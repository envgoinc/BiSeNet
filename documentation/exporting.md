# Exporting

The active path is:

```text
.pth checkpoint -> ONNX -> TensorRT engine
```

Do not treat LibTorch, NCNN, OpenVINO, or Triton folders as the main NV1 export path unless you have a specific reason.

## 1. Pick A Checkpoint

Start from a selected `.pth` checkpoint under `experiments/`.

Choose it based on multiple signals, not just the final checkpoint or one metric.

## 2. Make A Temporary Export Config

Before ONNX export, make a temporary copy of the training config and change the export resolution:

```python
cropsize = [896, 1184]
```

Use the target compile shape for your deployment. Keep the TensorRT shape aligned with this value.

Do not permanently change the training config just to export a model.

## 3. Convert `.pth` To ONNX

Use:

```text
tools/conversions/convert_pth_to_onnx.py
```

Command:

```bash
python tools/conversions/convert_pth_to_onnx.py --config /path/to/temp_export_config.py --weight-path /path/to/model.pth --outpath model.onnx
```

The script builds dummy input from `cfg.cropsize` and exports an ONNX model with input name `input_image`.

## 4. Compile ONNX To TensorRT

Run this on the target machine:

```bash
trtexec --onnx=/home/envgo/aquapilot/segmentation_cpp/models/one_shot_seg_11_17.onnx \
  --saveEngine=/home/envgo/aquapilot/segmentation_cpp/models/one_shot_seg_11_17.trt \
  --minShapes=input_image:1x3x896x1184 \
  --optShapes=input_image:1x3x896x1184 \
  --maxShapes=input_image:1x3x896x1184 \
  --fp16
```

If you change `cropsize`, update all three TensorRT shape arguments.

## Other Scripts

`tools/export_libtorch.py` exists, but it is not the documented NV1 path. It traces with a hardcoded dummy input of `1 x 3 x 1024 x 2048`.
