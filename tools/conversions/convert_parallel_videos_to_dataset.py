#!/usr/bin/env python3
from pathlib import Path
import cv2

# =========================
# Hardcoded config
# =========================
INPUT_VIDEO_IMAGES = "/app/birdseye_run_9/output_rgb.mp4"
INPUT_VIDEO_LABELS = "/app/birdseye_run_9/output_mask.mp4"

OUTPUT_IMAGES_DIR = "/app/birdseye_run_9/images"
OUTPUT_LABELS_DIR = "/app/birdseye_run_9/labels"

IMAGE_EXT = ".png"
LABEL_EXT = ".png"

FILENAME_PREFIX = "frame_"
ZERO_PAD = 6

REQUIRE_SAME_LENGTH = True
REQUIRE_SAME_RESOLUTION = False

PNG_COMPRESSION = 3
JPEG_QUALITY = 95

TARGET_FPS = None #IMPT set to None for exact decoding based on recorded frame rate


def imwrite_checked(path: Path, image) -> None:
    ext = path.suffix.lower()
    if ext == ".png":
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    elif ext in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        raise ValueError(f"Unsupported output extension: {ext}")
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def get_video_info(cap: cv2.VideoCapture, name: str) -> dict:
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {name}")
    return {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
    }


def main() -> None:
    output_images = Path(OUTPUT_IMAGES_DIR)
    output_labels = Path(OUTPUT_LABELS_DIR)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    cap_images = cv2.VideoCapture(INPUT_VIDEO_IMAGES)
    cap_labels = cv2.VideoCapture(INPUT_VIDEO_LABELS)

    try:
        info_images = get_video_info(cap_images, INPUT_VIDEO_IMAGES)
        info_labels = get_video_info(cap_labels, INPUT_VIDEO_LABELS)

        source_fps = info_images["fps"]

        print("Images video:")
        print(f"  frames={info_images['frame_count']} "
              f"size={info_images['width']}x{info_images['height']} "
              f"fps={source_fps}")
        print("Labels video:")
        print(f"  frames={info_labels['frame_count']} "
              f"size={info_labels['width']}x{info_labels['height']} "
              f"fps={info_labels['fps']}")

        # ── Compute frame-skip interval ──────────────────────────
        if TARGET_FPS is None:
            frame_interval = 1
            print(f"Extracting all frames at source FPS ({source_fps})")
        else:
            if TARGET_FPS > source_fps:
                raise ValueError(
                    f"TARGET_FPS ({TARGET_FPS}) exceeds source FPS ({source_fps})")
            frame_interval = max(1, round(source_fps / TARGET_FPS))
            effective_fps  = source_fps / frame_interval
            print(f"TARGET_FPS={TARGET_FPS} → keeping every {frame_interval}th frame "
                  f"(effective {effective_fps:.2f} fps)")

        if REQUIRE_SAME_RESOLUTION:
            if (info_images["width"]  != info_labels["width"] or
                info_images["height"] != info_labels["height"]):
                raise RuntimeError(
                    f"Resolution mismatch: "
                    f"{info_images['width']}x{info_images['height']} vs "
                    f"{info_labels['width']}x{info_labels['height']}")

        if REQUIRE_SAME_LENGTH and \
                info_images["frame_count"] != info_labels["frame_count"]:
            raise RuntimeError(
                f"Frame count mismatch: "
                f"{info_images['frame_count']} vs {info_labels['frame_count']}")

        src_idx   = 0   # source frame counter (every frame read from video)
        out_idx   = 0   # output frame counter (only saved frames)

        while True:
            ok_img, frame_img = cap_images.read()
            ok_lbl, frame_lbl = cap_labels.read()

            if not ok_img and not ok_lbl:
                break

            if REQUIRE_SAME_LENGTH and ok_img != ok_lbl:
                raise RuntimeError(
                    f"Videos misaligned at source frame {src_idx}.")

            if not ok_img or not ok_lbl:
                print(f"Stopping at source frame {src_idx}: one video ended.")
                break

            # ── Skip frames that don't fall on the target interval ──
            if src_idx % frame_interval == 0:
                stem         = f"{FILENAME_PREFIX}{out_idx:0{ZERO_PAD}d}"
                out_img_path = output_images / f"{stem}{IMAGE_EXT}"
                out_lbl_path = output_labels / f"{stem}{LABEL_EXT}"
                imwrite_checked(out_img_path, frame_img)
                imwrite_checked(out_lbl_path, frame_lbl)
                out_idx += 1

                if out_idx % 100 == 0:
                    print(f"Wrote {out_idx} frame pairs "
                          f"(source frame {src_idx})...")

            src_idx += 1

        print(f"\nDone. Read {src_idx} source frames, wrote {out_idx} frame pairs.")
        print(f"Images saved to: {output_images}")
        print(f"Labels saved to: {output_labels}")

    finally:
        cap_images.release()
        cap_labels.release()


if __name__ == "__main__":
    main()