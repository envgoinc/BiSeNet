# Getting Started And Commands

Use this as the main project flow. Some datasets skip the video or sim-mask steps.

## Normal Workflow

1. Set up the environment and dataset mounts.
2. Convert parallel videos to image/label folders if needed.
3. Clean sim masks if needed.
4. Generate split JSON from parallel image/label folders.
5. Update `configs/bisenetv2_envgo_dataset2.py`.
6. Train with `torchrun`.
7. Inspect `experiments/` outputs and checkpoints.
8. Evaluate relative model performance.
9. Make a temporary export config with the target export resolution.
10. Convert the selected `.pth` checkpoint to ONNX.
11. Compile the ONNX to TensorRT on the target machine.

## 1. Convert Videos If Needed

Use this only when starting from parallel image and label videos:

```text
tools/conversions/convert_parallel_videos_to_dataset.py
```

Edit the constants at the top, then run:

```bash
python tools/conversions/convert_parallel_videos_to_dataset.py
```

This is intended for constant-frame-rate parallel videos. Check output quality before using the frames for training.

## 2. Clean Sim Masks If Needed

For sim masks, run this before split generation:

```text
tools/sim_data_tools/boat_mask_addition.py
```

Edit the input labels directory, output labels directory, and reference mask path, then run:

```bash
python tools/sim_data_tools/boat_mask_addition.py
```

## 3. Generate Split JSON

Use:

```text
tools/splits/generate_new_splits.py
```

Edit these constants at the top:

```python
IMAGES_DIR = "/app/birdseye_run_12/images"
LABELS_DIR = "/app/birdseye_run_12/labels"
OUTPUT_JSON = "/app/birdseye_run_12/generated_splits.json"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
STRICT_LABELS = True
MIN_BINS_PER_SPLIT = 10
MAX_BIN_LEN = 80
```

Run:

```bash
python tools/splits/generate_new_splits.py
```

Optional checks:

```bash
python tools/splits/print_split_counts.py /path/to/generated_splits.json
python tools/splits/fix_splits_json.py --images-dir /path/to/images --labels-dir /path/to/labels --in-json /path/to/splits.json --out-json /path/to/fixed_splits.json --strict-labels
```

## 4. Update The Config

Use:

```text
configs/bisenetv2_envgo_dataset2.py
```

Set:

```python
dataset = "MultiSourceJsonDataset"
datasets = [
    {"images_dir": "...", "labels_dir": "...", "annpath": "..."},
]
```

Each entry should point to a prepared dataset source and its split JSON.

## 5. Train

Use the training entry point that exists in this checkout:

```text
tools/train_amp.py
```

Even though some internal notes refer to `tools/train.py`, that file is not present here. Use the file that exists unless a newer branch restores `tools/train.py`.

Single GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train_amp.py --config configs/bisenetv2_envgo_dataset2.py
```

Multiple GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train_amp.py --config configs/bisenetv2_envgo_dataset2.py
```

Fine-tune from a checkpoint:

```bash
torchrun --nproc_per_node=1 tools/train_amp.py --config configs/bisenetv2_envgo_dataset2.py --finetune-from /path/to/model.pth
```

## 6. Evaluate

```bash
python tools/evaluate.py --config configs/bisenetv2_envgo_dataset2.py --weight-path /path/to/model.pth
```

Distributed evaluation:

```bash
torchrun --nproc_per_node=2 tools/evaluate.py --config configs/bisenetv2_envgo_dataset2.py --weight-path /path/to/model.pth
```

These metrics are for relative comparison. Use a separate true evaluation dataset for final judgment.

## 7. Export ONNX

Make a temporary config with the export crop size, for example:

```python
cropsize = [896, 1184]
```

Then run:

```bash
python tools/conversions/convert_pth_to_onnx.py --config /path/to/temp_export_config.py --weight-path /path/to/model.pth --outpath model.onnx
```

Compile the ONNX to TensorRT on the target machine. See [Exporting](exporting.md).
