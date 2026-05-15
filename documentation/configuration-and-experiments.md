# Configuration And Experiments

For normal NV1 work, use:

```text
configs/bisenetv2_envgo_dataset2.py
```

Do not treat every config in `configs/` as active. Many are inherited or older experiments.

## Dataset Loader

Use:

```python
dataset = "MultiSourceJsonDataset"
```

This is the important loader for the current workflow. It lets one training run combine multiple prepared datasets.

The main config uses this structure:

```python
datasets = [
    {
        "images_dir": "/app/mar23rd_data/merged/images",
        "labels_dir": "/app/mar23rd_data/merged/labels_png_coloured",
        "annpath": "/app/mar23rd_data/merged/merged_splits_cleaned.json",
    },
    {
        "images_dir": "/app/BiSeNet/data_birdseye_long_selection/images",
        "labels_dir": "/app/BiSeNet/data_birdseye_long_selection/labels_coloured",
        "annpath": "/app/BiSeNet/data_birdseye_long_selection/matched_splits.fixed2.json",
    },
    {
        "images_dir": "/app/birdseye_run_10/images",
        "labels_dir": "/app/birdseye_run_10/labels_cleaned",
        "annpath": "/app/birdseye_run_10/generated_splits.json",
    },
    {
        "images_dir": "/app/birdseye_run_11/images",
        "labels_dir": "/app/birdseye_run_11/labels_cleaned",
        "annpath": "/app/birdseye_run_11/generated_splits.json",
    },
    {
        "images_dir": "/app/birdseye_run_12/images",
        "labels_dir": "/app/birdseye_run_12/labels_cleaned",
        "annpath": "/app/birdseye_run_12/generated_splits.json",
    },
]
```

Commented dataset entries are historical or optional context. Do not enable them unless you know why.

## Key Training Fields

Keep these unless you are intentionally changing the model or data contract:

```python
model_type = "bisenetv2"
n_cats = 3
num_aux_heads = 4
dataset = "MultiSourceJsonDataset"
lb_ignore = 255
```

`num_aux_heads=4` should stay unless there is a deliberate architecture change.

`lb_ignore=255` is a uint8 class ignore value. It is not a mask color to ignore.

The RGB mask color map is:

```python
label_color_map = [
    [255, 0, 0],   # self
    [0, 255, 0],   # objects
    [0, 0, 255],   # water
]
```

## Tuning Fields

These usually need judgment:

```python
lr_start = 1e-3
max_iter = 450000
scales = [0.75, 2.0]
cropsize = [512, 512]
eval_crop = [512, 512]
eval_scales = [0.9, 1.0, 1.75]
```

Tune `max_iter` based on dataset size, dataset complexity, and augmentations. Learning rate is similar. Metrics can mislead, so avoid treating one number as the whole decision.

For training, `cropsize=[512, 512]` was used. For export and TensorRT compile, use a temporary config with `cropsize=[896, 1184]` or the target compile shape.

Crop size is not just input size. It controls how much image context the model sees during training. Pair it with `scales`.

Rough training image flow:

1. Resize within the configured scale bounds, such as `0.75` to `2.0`.
2. Take a random `512 x 512` crop.
3. Apply augmentations.
4. Pass the crop to the model.

## W&B Fields

W&B is configured in the training config:

```python
wandb = True
wandb_project = "bisenetv2-BEV"
wandb_entity = None
wandb_run_name = "new_trimmed + new sim data"
wandb_notes = "augmentations included"
wandb_tags = ["amp", "ddp"]
```

For a new run, update `wandb_run_name` and `wandb_notes` so they describe changes in dataset mix, config, augmentations, weights, or checkpoint source.

`tools/train_amp.py` initializes W&B only on rank 0 and logs training loss, validation loss, learning rate, GPU memory, and final evaluation metrics.

## Experiment Outputs

Training writes to:

```text
experiments/<config_name>/<timestamp>-<wandb_run_name>/
```

Each job should include:

- a copied config under `logs/training_parameters.py`
- training checkpoints under `models/training_checkpoints/`
- a final checkpoint under `models/`

Use the copied config to confirm what actually ran.

Choose checkpoints using a range of evidence: validation curves, relative post-training metrics, performance across conditions, out-of-distribution data, and deployment behavior when available.

Do not assume the newest checkpoint or best single metric is the right model.

For context on why training choices were made, ask Brendan.
