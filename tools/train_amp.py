#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, log_msg

from tqdm import tqdm
import wandb
from datetime import datetime
from pathlib import Path


## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)
cfg_name = osp.splitext(osp.basename(args.config))[0]
exp_dir = None
logs_dir = None
models_dir = None
ckpt_dir = None
run_id_base = None  # used for filenames



def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        msg = net.load_state_dict(torch.load(args.finetune_from,
            map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [OhemCELoss(0.7, lb_ignore)
            for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        find_unused_parameters=True,
        output_device=local_rank
        )
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def is_main_process(): # useful for wandb to not log mutiple times if running across mult. processes
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_mean_scalar(x: float, device="cuda"): #useful for wandb 
    """Average a python float across ranks (DDP)."""
    if (not dist.is_available()) or (not dist.is_initialized()):
        return x
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

def wandb_init(cfg): #wandb initialization, pull from config
    """Initialize wandb only on rank 0."""
    if not is_main_process():
        return None

    use_wandb = getattr(cfg, "wandb", False) if not isinstance(cfg, dict) else cfg.get("wandb", False)
    if not use_wandb:
        return None

    # cfg may be dict-like or attr-like depending on your set_cfg_from_file
    def _get(k, default=None):
        if isinstance(cfg, dict): return cfg.get(k, default)
        return getattr(cfg, k, default)

    run = wandb.init(
        project=_get("wandb_project", f"{_get('model_type','model')}-{_get('dataset','dataset')}"),
        entity=_get("wandb_entity", None),
        name=_get("wandb_run_name", None),
        tags=_get("wandb_tags", None),
        notes=_get("wandb_notes", None),
        config=cfg.__dict__
    )
    return run

def train():
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train') #data loader
    vdl = get_data_loader(cfg, mode='val') #validation data loader
    lb_ignore = getattr(cfg, "lb_ignore", 255)


    ## model
    net, criteria_pre, criteria_aux = set_model(lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    pbar = dl
    if is_main_process():
        pbar = tqdm(dl, total=cfg.max_iter, dynamic_ncols=True)

    for it, (im, lb) in enumerate(pbar):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)
        optim.zero_grad(set_to_none=True) #BC: added set to none, its faster and doesn't break anything
        with amp.autocast(enabled=cfg.use_fp16):
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        # checkpointing (rank0 only)
        ckpt_every = getattr(cfg, "checkpoint_interval", 0)
        if ckpt_every and (it + 1) % ckpt_every == 0 and dist.get_rank() == 0:
            ckpt_path = osp.join(ckpt_dir, f"{it+1}.pth")
            torch.save(net.module.state_dict(), ckpt_path)



        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        #statistics logging
        if (it + 1) % 20 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)

            loss_avg = ddp_mean_scalar(loss.item())
            loss_pre_avg = ddp_mean_scalar(loss_pre.item())
            loss_aux_avg = [ddp_mean_scalar(l.item()) for l in loss_aux] if len(loss_aux) else []

            if is_main_process():
                log_dict = {
                    "iter/train": it + 1,
                    "lr/train": lr,
                    "loss/train": loss_avg,
                    "loss_pre/train": loss_pre_avg,
                    "iter_time/train": time_meter.get() if hasattr(time_meter, "get") else None,
                    "gpu/mem_alloc_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }

                for i, la in enumerate(loss_aux_avg):
                    log_dict[f"loss_aux{i}"] = la

                # remove None values (in case time_meter.get doesn't exist)
                log_dict = {k: v for k, v in log_dict.items() if v is not None}

                if wandb.run is not None:
                    wandb.log(log_dict, step=it + 1)

                if isinstance(pbar, tqdm):
                    # display key metrics nicely
                    pbar.set_postfix({
                        "lr": f"{lr:.3e}",
                        "loss": f"{loss_avg:.4f}",
                        "pre": f"{loss_pre_avg:.4f}",
                    })

        ## print training log message
        if (it + 1) % 1000 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            msg = log_msg(
                it, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)
            logger.info(msg)

        # --- validation logging ---
        val_every = getattr(cfg, "val_interval", 0)
        if val_every and (it + 1) % val_every == 0:
            net.eval()
            # if your model has aux_mode like in evaluator, keep it consistent
            org_aux = getattr(net.module, "aux_mode", None)
            if org_aux is not None:
                net.module.aux_mode = "eval"

            v_loss_sum = 0.0
            v_pre_sum = 0.0
            v_aux_sums = [0.0 for _ in range(cfg.num_aux_heads)]
            v_n = 0

            max_vb = int(getattr(cfg, "val_num_batches", 0) or 0)

            with torch.no_grad():
                for vb, (vim, vlb) in enumerate(vdl):
                    if max_vb > 0 and vb >= max_vb:
                        break

                    vim = vim.cuda()
                    vlb = vlb.cuda()
                    vlb = torch.squeeze(vlb, 1)

                    with amp.autocast(enabled=cfg.use_fp16):
                        vlogits, *vlogits_aux = net(vim)
                        vloss_pre = criteria_pre(vlogits, vlb)
                        vloss_aux = [crit(lgt, vlb) for crit, lgt in zip(criteria_aux, vlogits_aux)]
                        vloss = vloss_pre + sum(vloss_aux)

                    v_loss_sum += float(vloss.item())
                    v_pre_sum += float(vloss_pre.item())
                    for i, la in enumerate(vloss_aux):
                        v_aux_sums[i] += float(la.item())
                    v_n += 1

            # averages on each rank
            v_loss_avg = v_loss_sum / max(v_n, 1)
            v_pre_avg = v_pre_sum / max(v_n, 1)
            v_aux_avgs = [s / max(v_n, 1) for s in v_aux_sums]

            # average across ranks
            v_loss_avg = ddp_mean_scalar(v_loss_avg)
            v_pre_avg = ddp_mean_scalar(v_pre_avg)
            v_aux_avgs = [ddp_mean_scalar(x) for x in v_aux_avgs]

            if is_main_process():
                val_log = {
                    "iter": it + 1,
                    "loss/val": v_loss_avg,
                    "loss_pre/val": v_pre_avg,
                }
                for i, va in enumerate(v_aux_avgs):
                    val_log[f"val/loss_aux{i}"] = va

                if wandb.run is not None:
                    wandb.log(val_log, step=it + 1)

            # restore model state
            if org_aux is not None:
                net.module.aux_mode = org_aux
            net.train()

        lr_schdr.step()

    ## dump the final model and evaluate the result
    save_pth = osp.join(models_dir, f"{run_id_base}.pth")
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)


    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)

    if is_main_process() and wandb.run is not None:
        # Convert tabulate-style outputs into dict metrics if possible
        # f1_content and iou_content look like rows; assume first column is class name
        def table_to_metrics(heads, content, prefix):
            metrics = {}
            if len(content) == 0:
                return metrics
            # try to infer format:
            # heads: ["class", "f1", ...], content: [["bg", 0.9, ...], ...]
            for row in content:
                if len(row) != len(heads):
                    continue
                class_name = str(row[0])
                for h, v in zip(heads[1:], row[1:]):
                    try:
                        metrics[f"{prefix}/{class_name}/{h}"] = float(v)
                    except:
                        pass
            return metrics

        f1_metrics = table_to_metrics(f1_heads, f1_content, "eval/f1")
        iou_metrics = table_to_metrics(iou_heads, iou_content, "eval/iou")

        wandb.log({**f1_metrics, **iou_metrics}, step=cfg.max_iter)

    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

    return


def main():
    global exp_dir, logs_dir, models_dir, ckpt_dir, run_id_base

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    exp_name = getattr(cfg, "wandb_run_name", "unnamed_exp")
    stamp = datetime.now().strftime("%b_%d_%Y_%H:%M")  # same stamp used for folder
    exp_dir = osp.join("experiments", cfg_name, f"{stamp}-{exp_name}")
    logs_dir = osp.join(exp_dir, "logs")

    models_dir = osp.join(exp_dir, "models")
    ckpt_dir = osp.join(models_dir, "training_checkpoints")

    exp_name_file = str(exp_name).replace(" ", "_").replace("-", "_")
    run_id_base = f"{cfg_name}_{stamp}_{exp_name_file}" 

    if dist.get_rank() == 0:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        dst_cfg = osp.join(logs_dir, "training_parameters.py")
        header = "# THIS IS A COPY OF THE PARAMETER STATE WHEN THE MODEL WAS TRAINED; DO NOT MODIFY\n\n"
        src_text = Path(args.config).read_text(encoding="utf-8")
        Path(dst_cfg).write_text(header + src_text, encoding="utf-8")


    dist.barrier()

    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', logs_dir)

    run = wandb_init(cfg)
    try:
        train()
    finally:
        if run is not None:
            run.finish()



if __name__ == "__main__":
    main()
