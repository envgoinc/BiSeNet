# Setup

This repo does not currently include a `requirements.txt`, `pyproject.toml`, `environment.yml`, Dockerfile, or Makefile.

TODO(owner): add the supported Python, CUDA, PyTorch, and package versions for NV1 training.

Known imports used by the active path:

- `torch`
- `opencv-python` / `cv2`
- `numpy`
- `tqdm`
- `tabulate`
- `wandb`

Historical environment notes from the old README:

- Ubuntu 18.04
- CUDA 10.2 or 11.3
- cuDNN 8
- Python 3.8.8
- PyTorch 1.11.0

Treat those as historical until the current training environment is confirmed.

## Paths

Most active examples use `/app/...` paths. Either mount datasets at those paths inside the training container or update `configs/bisenetv2_envgo_dataset2.py`.

Before training, all `images_dir`, `labels_dir`, and `annpath` values in the config must exist on the training machine.

TODO(owner): add the exact environment creation and install commands.
