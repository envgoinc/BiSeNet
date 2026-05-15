# Training And Evaluation

Training starts after the split JSON exists and `configs/bisenetv2_envgo_dataset2.py` points at the right dataset sources.

## Active Entry Point

Use:

```text
tools/train_amp.py
```

This file exists in the checkout and initializes distributed training from `LOCAL_RANK`. Run it with `torchrun`.

Some notes mention `tools/train.py`, but that file is not present here. Do not use a non-existent path just because an old note says so.

## Training Commands

Single GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp.py --config configs/bisenetv2_envgo_dataset2.py
```

Two GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train_amp.py --config configs/bisenetv2_envgo_dataset2.py
```

Fine-tune:

```bash
torchrun --nproc_per_node=1 tools/train_amp.py --config configs/bisenetv2_envgo_dataset2.py --finetune-from /path/to/model.pth
```

## What The Training Config Controls

The main config is:

```text
configs/bisenetv2_envgo_dataset2.py
```

Important fields:

```python
model_type = "bisenetv2"
n_cats = 3
num_aux_heads = 4
lr_start = 1e-3
max_iter = 450000
dataset = "MultiSourceJsonDataset"
cropsize = [512, 512]
scales = [0.75, 2.0]
```

Keep `num_aux_heads=4` unless you are intentionally changing the architecture.

Tune `max_iter` and `lr_start` based on dataset size, dataset complexity, augmentations, and observed training behavior.

## Crop Size During Training

Training used:

```python
cropsize = [512, 512]
```

This is not just input size. It controls how much image context the model sees.

The rough training image flow is:

1. Resize the image within `scales`, such as `0.75` to `2.0`.
2. Take a random `512 x 512` crop.
3. Apply augmentations.
4. Pass the crop to the model.

Setting crop size to the full image size is not automatically better. Pair crop size with scale settings.

## Experiment Outputs

Training writes to:

```text
experiments/<config_name>/<timestamp>-<wandb_run_name>/
```

Expected contents:

- `logs/training_parameters.py`: copied config from the run.
- `models/training_checkpoints/`: periodic checkpoints.
- `models/`: final checkpoint.

Use the copied config when comparing runs. It is the record of what actually trained.

## Evaluation

Run:

```bash
python tools/evaluate.py --config configs/bisenetv2_envgo_dataset2.py --weight-path /path/to/model.pth
```

`tools/evaluate.py` uses the config's `test` split through the dataset loader. In this repo, that is for relative post-training metrics.

Do not choose a checkpoint based only on the newest file or the best single validation number. Compare validation curves, relative evaluation metrics, conditions represented in the data, out-of-distribution behavior, and deployment behavior when available.

For final evaluation, use the separate true evaluation dataset outside the train/val/test split.

For more context on tuning choices, ask Brendan.
