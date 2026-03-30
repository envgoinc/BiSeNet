import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
import torch.distributed as dist
import lib.data.transform_cv2 as T
from lib.data.sampler import RepeatedDistSampler
from lib.data.cityscapes_cv2 import CityScapes
from lib.data.coco import CocoStuff
from lib.data.ade20k import ADE20k
from lib.data.customer_dataset import CustomerDataset
from lib.data.mastr import MASTR1325
from lib.data.BEV_long_sel import BEV_long_sel
from lib.data.fisheye_small import fisheye_small
from lib.data.merged_json_dataset import MergedJsonDataset
from lib.data.multi_source_dataset import MultiSourceJsonDataset

DATASET_REGISTRY = {
    "CityScapes":            CityScapes,
    "CocoStuff":             CocoStuff,
    "ADE20k":                ADE20k,
    "CustomerDataset":       CustomerDataset,
    "MASTR1325":             MASTR1325,
    "BEV_long_sel":          BEV_long_sel,
    "fisheye_small":         fisheye_small,
    "MergedJsonDataset":     MergedJsonDataset,
    "MultiSourceJsonDataset": MultiSourceJsonDataset,
}


def get_data_loader(cfg, mode='train'):
    if mode == 'train':
        trans_func = T.TransformationTrain(cfg.scales, cfg.cropsize)
        batchsize  = cfg.ims_per_gpu
        shuffle    = True
        drop_last  = True
    elif mode == 'val':
        trans_func = T.TransformationVal()
        batchsize  = cfg.eval_ims_per_gpu
        shuffle    = False
        drop_last  = False
    elif mode == 'test':
        trans_func = T.TransformationVal()
        batchsize  = cfg.eval_ims_per_gpu
        shuffle    = False
        drop_last  = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Resolve dataset class from registry — never hardcoded, never eval()
    dataset_name = getattr(cfg, "dataset", None)
    if dataset_name is None:
        raise ValueError("cfg.dataset must be set (e.g. 'MergedJsonDataset')")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"cfg.dataset='{dataset_name}' not found in DATASET_REGISTRY.\n"
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    dataset_cls = DATASET_REGISTRY[dataset_name]

    ds_specs = getattr(cfg, "datasets", None)
    if ds_specs is not None:
        # ---- Multi-dataset path ----
        if not isinstance(ds_specs, (list, tuple)) or len(ds_specs) == 0:
            raise ValueError("cfg.datasets must be a non-empty list of dataset specs")
        dsets = []
        for s in ds_specs:
            dsets.append(
                dataset_cls(
                    cfg=cfg,
                    images_dir=s["images_dir"],
                    labels_dir=s["labels_dir"],
                    annpath=s["annpath"],
                    trans_func=trans_func,
                    mode=mode,
                )
            )
        ds = dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)
    else:
        # ---- Legacy single-dataset path ----
        if mode == 'train':
            annpath = cfg.train_im_anns
        elif mode == 'val':
            annpath = cfg.val_im_anns
        else:
            annpath = cfg.test_im_anns
        ds = dataset_cls(cfg.im_root, annpath, trans_func=trans_func, mode=mode)

    # ---- Distributed vs non-distributed loader ----
    if dist.is_initialized():
        assert dist.is_available(), "dist should be initialized"
        if mode == 'train':
            assert cfg.max_iter is not None
            n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl