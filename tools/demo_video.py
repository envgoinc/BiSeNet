import sys
sys.path.insert(0, '.')
import argparse
import time
import threading
import queue
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True  # pick best conv algos

# ------------------ args ------------------
parse = argparse.ArgumentParser()
parse.add_argument('--config', type=str, default='configs/bisenetv2.py')
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth')
parse.add_argument('--input', type=str, default='./example.mp4')
parse.add_argument('--output', type=str, default='./res.mp4')
parse.add_argument('--batch-size', type=int, default=8)
parse.add_argument('--size', type=int, default=512)  # must be divisible by 32
parse.add_argument('--fp16', action='store_true', help='enable mixed precision')
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ model ------------------
def get_model():
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    state = torch.load(args.weight_path, map_location='cpu')
    net.load_state_dict(state, strict=False)
    net.eval().to(DEVICE)
    return net

# ------------------ threads & queues ------------------
read_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)   # raw BGR frames
write_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)  # colorized BGR frames
stop_token = object()

def reader(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        read_q.put(stop_token); return
    # Read basic props once
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = (w, h, fps)
    read_q.put(meta)  # first item: metadata

    while True:
        ret, frame = cap.read()
        if not ret: break
        read_q.put(frame)  # BGR uint8 HxWx3
    cap.release()
    read_q.put(stop_token)

def writer(path):
    meta = read_q.get()
    if meta is stop_token:
        return
    w, h, fps = meta
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Build a single deterministic BGR palette once
    palette_bgr = np.zeros((256, 3), dtype=np.uint8)

    palette_bgr[0] = [37, 195,  247]  # Class 0
    palette_bgr[1] = [ 224, 167, 41]  # Class 1
    palette_bgr[2] = [ 164,  75, 90]  # Class 2
    palette_bgr[3:] = [128, 128, 128]
    # palette_bgr = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    while True:
        item = write_q.get()
        if item is stop_token:
            break
        # item is either (B,H,W) labels or (H,W) single label map
        if item.ndim == 3:
            # batch of labels
            for lab in item:                    # (H,W) uint8
                frame_bgr = palette_bgr[lab]    # (H,W,3) uint8
                vw.write(frame_bgr)
        elif item.ndim == 2:
            frame_bgr = palette_bgr[item]
            vw.write(frame_bgr)
        else:
            # ignore unexpected
            pass

    vw.release()

# ------------------ preprocessing ------------------
def preprocess_batch(frames_bgr, out_h, out_w, mean, std):
    """
    frames_bgr: List[np.ndarray HxWx3 BGR uint8]
    Returns: torch.FloatTensor (B,3,out_h,out_w) normalized, pinned
    """
    # Resize on CPU via OpenCV (fast), convert to RGB, normalize
    batch = []
    for f in frames_bgr:
        r = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        r = r.astype(np.float32) / 255.0
        r = (r - mean) / std
        # HWC -> CHW
        r = np.transpose(r, (2, 0, 1))
        batch.append(r)
    arr = np.stack(batch, axis=0)   # (B,3,H,W)
    tensor = torch.from_numpy(arr).pin_memory()
    return tensor

# ------------------ inference loop ------------------
def main():
    net = get_model()

    # Optional FP16 (mixed precision). Safest via autocast; faster if model tolerates.
    use_amp = bool(args.fp16)

    # Cityscapes mean/std used by the original code
    mean = np.array([0.3257, 0.3690, 0.3223], dtype=np.float32)
    std  = np.array([0.2112, 0.2148, 0.2115], dtype=np.float32)

    # Kick off threads
    t_reader = threading.Thread(target=reader, args=(args.input,), daemon=True)
    t_writer = threading.Thread(target=writer, args=(args.output,), daemon=True)
    t_reader.start()
    t_writer.start()

    # Grab meta put by reader (writer already consumed its copy)
    meta = None
    while meta is None:
        x = read_q.get()
        if isinstance(x, tuple):
            meta = x
        elif x is stop_token:
            write_q.put(stop_token); return
        else:
            # Unexpected early frame; requeue it for processing
            read_q.put(x)
            break
    w, h, fps = meta if meta else (None, None, None)

    B = args.batch_size
    target = (args.size, args.size)

    # Allocate a CUDA stream to overlap H2D + compute
    stream = torch.cuda.Stream(device=DEVICE) if DEVICE.type == 'cuda' else None

    # Inference loop
    pending = []
    orig_sizes = []  # (H,W) per frame to restore
    frames_out = []

    def flush():
        """Run inference on accumulated 'pending' frames and enqueue results."""
        nonlocal pending, orig_sizes
        if not pending:
            return

        inp = preprocess_batch(
            pending, out_h=target[0], out_w=target[1], mean=mean, std=std
        )
        HsWs = orig_sizes[:]  # copy
        pending.clear()
        orig_sizes.clear()

        # Move to device (non_blocking with pinned host mem)
        if DEVICE.type == 'cuda':
            with torch.cuda.stream(stream):
                inp = inp.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = net(inp)[0]
                # Argmax on device for less PCIe traffic
                pred_small = logits.argmax(dim=1)  # (B,h,w)
            # Sync before using results on CPU
            stream.synchronize()
        else:
            with torch.cuda.amp.autocast(enabled=False):
                logits = net(inp)[0]
            pred_small = logits.argmax(dim=1)

        # Upsample each mask back to original frame size (CPU or GPU then CPU)
        preds = pred_small.detach()
        # Use bilinear upsample on probs usually; but we only have labels now.
        # We'll upsample with nearest to keep labels intact.
        # if DEVICE.type == 'cuda':
        #     preds = F.interpolate(
        #         preds.unsqueeze(1).float(),
        #         size=None, scale_factor=None, mode='nearest',
        #         # We'll handle per-frame resize below to exact H,W for speed
        #     ).squeeze(1).to('cpu', non_blocking=False)

        # preds = preds.cpu().numpy().astype(np.uint8)  # (B,h,w) labels

        preds = pred_small.detach().to('cpu', non_blocking=False).numpy().astype(np.uint8)  # (B,h,w)

        # Resize each label map back to original frame size with nearest
        resized_labels = []
        for lab, (Ho, Wo) in zip(preds, HsWs):
            lab = cv2.resize(lab, (Wo, Ho), interpolation=cv2.INTER_NEAREST)
            resized_labels.append(lab)

        # Send LABELS to writer (let writer colorize with a single, consistent palette)
        write_q.put(np.stack(resized_labels, axis=0))  # (B,H,W) uint8

    # Consume frames from reader and batch them
    while True:
        item = read_q.get()
        if item is stop_token:
            break
        if isinstance(item, tuple):
            # already handled meta above; ignore if seen again
            continue
        frame = item  # BGR uint8
        pending.append(frame)
        orig_sizes.append((frame.shape[0], frame.shape[1]))
        if len(pending) >= B:
            flush()

    # flush remainder
    flush()

    # tell writer to finish
    write_q.put(stop_token)

    # join threads
    t_reader.join()
    t_writer.join()

if __name__ == '__main__':
    main()
