# Data Preparation And Splits

The first real step after setup is usually creating a dataset split.

The active path starts from parallel image and label folders:

```text
images/ -> input frames
labels/ -> RGB masks
```

The split JSON tells the dataset loader which frame ranges belong to `train`, `val`, and `test`.

## Why Buckets Exist

Video data creates many near-duplicate frames. A 10 minute video can produce thousands of image/label pairs. If those frames are split randomly at frame level, very similar frames can land in train, val, and test.

To reduce leakage, split contiguous frame ranges as buckets. A dataset might become hundreds of contiguous segment buckets, then those buckets are assigned to train, val, and test.

In this checkout, `tools/splits/generate_new_splits.py` sorts frame pairs, applies `TRAIN_FRAC`, `VAL_FRAC`, and `TEST_FRAC`, then converts each split into contiguous buckets. It does not appear to randomly shuffle buckets before assignment. If random bucket assignment is required, change that script deliberately and review the output JSON.

Split meanings in this repo:

- `train`: used for training.
- `val`: used for validation curves and training-time metrics.
- `test`: used for relative post-training metrics.

For true final evaluation, use a separate dataset outside this train/val/test split.

## If Starting From Videos

If you have parallel image and label videos with a constant frame rate, use:

```text
tools/conversions/convert_parallel_videos_to_dataset.py
```

Edit the hardcoded config at the top:

```python
INPUT_VIDEO_IMAGES = "/app/birdseye_run_9/output_rgb.mp4"
INPUT_VIDEO_LABELS = "/app/birdseye_run_9/output_mask.mp4"
OUTPUT_IMAGES_DIR = "/app/birdseye_run_9/images"
OUTPUT_LABELS_DIR = "/app/birdseye_run_9/labels"
TARGET_FPS = None
```

Keep `TARGET_FPS = None` if you want exact decoding at the recorded frame rate.

## Sim Mask Cleaning

For sim data, run mask cleaning before split generation:

```text
tools/sim_data_tools/boat_mask_addition.py
```

This converts sim masks into the RGB mask format used by training. It snaps mask pixels to expected colors and overlays the reference sticker.

Edit the hardcoded paths at the top:

```python
INPUT_LABELS_DIR = Path("/app/birdseye_run_10/labels")
OUTPUT_LABELS_DIR = Path("/app/birdseye_run_10/labels_cleaned")
BOAT_REFERENCE_MASK = Path("/app/BiSeNet/data_birdseye_long_selection/labels/bev_C_00094m.png")
```

## Generate The Split JSON

Use:

```text
tools/splits/generate_new_splits.py
```

This script has no CLI. Edit the constants at the top:

```python
IMAGES_DIR = "/app/birdseye_run_12/images"
LABELS_DIR = "/app/birdseye_run_12/labels"
OUTPUT_JSON = "/app/birdseye_run_12/generated_splits.json"

EXPERIMENT_NAME = "generated"
SOURCE_EXPERIMENTS = []

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

STRICT_LABELS = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_EXTS = {".png", ".jpg", ".jpeg"}

MIN_BINS_PER_SPLIT = 10
MAX_BIN_LEN = 80
```

Then run:

```bash
python tools/splits/generate_new_splits.py
```

The output is usually named `generated_splits.json`, but use whatever path is set in `OUTPUT_JSON`.

## Data Quality Rule

Images may be JPEG or PNG. Masks should strongly prefer PNG.

Do not add conversion steps that silently change compression, encoding, frame rate, image quality, PNG compression, JPEG quality, or what the model sees. That can create a train-to-deploy gap.

When adding or editing conversion scripts, check the codec, frame rate, and write parameters explicitly.

## Useful Split Checks

Count frames per split:

```bash
python tools/splits/print_split_counts.py /path/to/generated_splits.json
```

Repair a split against actual files:

```bash
python tools/splits/fix_splits_json.py --images-dir /path/to/images --labels-dir /path/to/labels --in-json /path/to/splits.json --out-json /path/to/fixed_splits.json --strict-labels
```
