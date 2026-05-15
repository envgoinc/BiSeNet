# BiSeNet NV1 Segmentation Training

This repo trains BiSeNet-based segmentation models for NV1.

It is not the full raw-data-to-model pipeline. Upstream dataset generation and automation mostly live elsewhere. This repo starts from prepared segmentation datasets, or from parallel image/label sources that need repo-compatible split JSON files.

Normal workflow:

```text
setup -> data split -> config -> training -> evaluation -> ONNX export -> TensorRT compile
```

For normal NV1 work, start from [`configs/bisenetv2_envgo_dataset2.py`](configs/bisenetv2_envgo_dataset2.py). That config uses `MultiSourceJsonDataset` to combine multiple prepared dataset sources in one training run.

Docs:

- [Setup](documentation/setup.md)
- [Getting started and commands](documentation/getting-started-and-general-commands.md)
- [Data preparation and splits](documentation/data-preparation-and-splits.md)
- [Configuration and experiments](documentation/configuration-and-experiments.md)
- [Training and evaluation](documentation/training-and-evaluation.md)
- [Exporting](documentation/exporting.md)

This repo has older configs, scripts, and deployment folders. Do not treat every file as active.
