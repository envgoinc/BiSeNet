#version 1  
# import os.path as osp
# import json
# import cv2
# import numpy as np

# import lib.data.transform_cv2 as T
# from torch.utils.data import Dataset


# class MergedJsonDataset(Dataset):
#     """
#     Uses a JSON spec file to expand frame ranges into (image,label) file pairs.

#     Expected directory layout under dataroot:
#       dataroot/
#         merged/
#           images/
#             image_<experiment>_frame000001.jpg
#             ...
#           labels/
#             label_<experiment>_frame000001.jpg
#             ...

#     JSON has:
#       experiment: "2025-10-23-12-56-47"
#       format: { images_prefix, labels_prefix, ext }
#       splits: { train: [...], val: [...], test: [...] }
#     """

#     def __init__(self, dataroot, annpath, trans_func=None, mode="train"):
#         assert mode in ("train", "val", "test")
#         self.mode = mode
#         self.trans_func = trans_func

#         # ignore index for segmentation losses
#         self.lb_ignore = 255

#         # TODO: set to correct number of classes once you confirm (binary vs 3-class etc.)
#         self.n_cats = 2

#         # match your other datasets
#         self.to_tensor = T.ToTensor(
#             mean=(0.4, 0.4, 0.4),
#             std=(0.2, 0.2, 0.2),
#         )

#         with open(annpath, "r") as f:
#             spec = json.load(f)

#         exp = spec["experiment"]
#         fmt = spec.get("format", {})
#         img_prefix = fmt.get("images_prefix", "image_")
#         lb_prefix = fmt.get("labels_prefix", "label_")
#         ext = fmt.get("ext", "jpg")

#         images_dir = osp.join(dataroot, "merged", "images")
#         labels_dir = osp.join(dataroot, "merged", "labels")

#         segments = spec["splits"][mode]

#         self.img_paths = []
#         self.lb_paths = []

#         for seg in segments:
#             start = int(seg["start_frame_num"])
#             end = int(seg["end_frame_num"])

#             for frame in range(start, end + 1):
#                 img_name = f"{img_prefix}{exp}_frame{frame:06d}.{ext}"
#                 lb_name = f"{lb_prefix}{exp}_frame{frame:06d}.{ext}"
#                 self.img_paths.append(osp.join(images_dir, img_name))
#                 self.lb_paths.append(osp.join(labels_dir, lb_name))

#         assert len(self.img_paths) == len(self.lb_paths)
#         self._len = len(self.img_paths)

#     def __len__(self):
#         return self._len

#     def __getitem__(self, idx):
#         impth = self.img_paths[idx]
#         lbpth = self.lb_paths[idx]

#         img = cv2.imread(impth)
#         if img is None:
#             raise FileNotFoundError(f"Image not found: {impth}")
#         img = img[:, :, ::-1].copy()  # BGR->RGB

#         lb = cv2.imread(lbpth, cv2.IMREAD_UNCHANGED)
#         if lb is None:
#             raise FileNotFoundError(f"Label not found: {lbpth}")

#         # Convert 3-channel label image -> single-channel class indices.
#         # TEMPORARY behavior: treat as binary where "anything bright" is foreground.
#         if lb.ndim == 3:
#             fg = (lb[:, :, 0] > 127) | (lb[:, :, 1] > 127) | (lb[:, :, 2] > 127)
#             label = fg.astype(np.uint8)  # 0 or 1
#         else:
#             label = (lb > 127).astype(np.uint8)

#         im_lb = {"im": img, "lb": label}

#         if self.trans_func is not None:
#             im_lb = self.trans_func(im_lb)

#         im_lb = self.to_tensor(im_lb)

#         # training code expects label shape [1,H,W] per sample
#         return im_lb["im"].detach(), im_lb["lb"].unsqueeze(0).detach()


# lib/data/merged_json_dataset.py
import os
import os.path as osp
import json
import re
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

import lib.data.transform_cv2 as T


class MergedJsonDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        annpath: str,
        trans_func=None,
        mode: str = "train",
    ):
        assert mode in ("train", "val", "test"), f"mode must be train/val/test, got {mode}"
        self.mode = mode
        self.trans_func = trans_func

        self.lb_ignore = 255
        self.n_cats = 2  # keep as-is; cfg.n_cats must match your decode

        self.to_tensor = T.ToTensor(mean=(0.4, 0.4, 0.4), std=(0.2, 0.2, 0.2))

        self.images_dir = osp.abspath(images_dir)
        self.labels_dir = osp.abspath(labels_dir)
        annpath = osp.abspath(annpath)

        if not osp.isdir(self.images_dir):
            raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
        if not osp.isdir(self.labels_dir):
            raise FileNotFoundError(f"labels_dir not found: {self.labels_dir}")
        if not osp.isfile(annpath):
            raise FileNotFoundError(f"annpath not found: {annpath}")

        with open(annpath, "r") as f:
            spec = json.load(f)

        if "splits" not in spec or mode not in spec["splits"]:
            raise KeyError(f"JSON spec missing splits['{mode}']")

        segments = spec["splits"][mode]
        if not isinstance(segments, list) or len(segments) == 0:
            raise ValueError(f"splits['{mode}'] is empty or not a list")

        self.img_paths = []
        self.lb_paths = []
        for seg in segments:
            self._append_segment(seg)

        if len(self.img_paths) != len(self.lb_paths):
            raise RuntimeError("img_paths and lb_paths length mismatch")
        if len(self.img_paths) == 0:
            raise RuntimeError("No samples found after expanding splits")

        self._len = len(self.img_paths)

    # ----------------------------
    # Path resolution / expansion
    # ----------------------------

    # @staticmethod
    # def _resolve_image_label_dirs(dataroot: str, json_dir: str) -> Tuple[str, str]:
    #     """
    #     Discover image/label directories without hardcoding a single expected layout.

    #     Tries these patterns (in order), for both dataroot and json_dir:
    #       1) <base>/images and <base>/labels
    #       2) <base>/merged/images and <base>/merged/labels
    #       3) <base>/merged/images and <base>/merged/labels (if json is under merged/)
    #       4) <base>/data/images and <base>/data/labels (optional common pattern)

    #     First pair where both directories exist is used.
    #     """
    #     bases = [dataroot, json_dir]

    #     candidates = []
    #     for base in bases:
    #         candidates.extend([
    #             (osp.join(base, "images"), osp.join(base, "labels")),
    #             (osp.join(base, "merged", "images"), osp.join(base, "merged", "labels")),
    #             (osp.join(base, "data", "images"), osp.join(base, "data", "labels")),
    #         ])

    #     for imd, lbd in candidates:
    #         if osp.isdir(imd) and osp.isdir(lbd):
    #             return imd, lbd

    #     # If nothing matched, provide a useful error.
    #     msg = [
    #         "Could not locate images/labels directories.",
    #         f"dataroot={dataroot}",
    #         f"annpath_dir={json_dir}",
    #         "Tried candidates:",
    #     ]
    #     for imd, lbd in candidates:
    #         msg.append(f"  - {imd}  AND  {lbd}")
    #     raise FileNotFoundError("\n".join(msg))

    def _candidate_label_paths(self, img_path: str) -> list[str]:
        """
        Given an image path, return candidate label paths (absolute) in priority order.
        This supports mixed naming schemes within the same dataset.
        """
        img_base = osp.basename(img_path)

        # 1) identical filename in labels dir
        cands = [osp.join(self.labels_dir, img_base)]

        # 2) insert 'm' before extension: bev_back_00001.jpg -> bev_back_00001m.png or .jpg
        stem, ext = osp.splitext(img_base)
        # try same extension first (some datasets use .jpg masks)
        cands.append(osp.join(self.labels_dir, f"{stem}m{ext}"))
        # common mask ext
        cands.append(osp.join(self.labels_dir, f"{stem}m.png"))

        # 3) optional legacy: image_* -> label_* (only if you actually have that)
        if stem.startswith("image_"):
            cands.append(osp.join(self.labels_dir, f"label_{stem[len('image_') :]}{ext}"))
            cands.append(osp.join(self.labels_dir, f"label_{stem[len('image_') : ]}.png"))

        return cands

    def _resolve_label_path(self, img_path: str, proposed_label_path: str | None) -> str:
        """
        Resolve the correct label path for an image. If proposed_label_path exists, use it;
        otherwise try known schemes and pick the first existing file.
        """
        if proposed_label_path and osp.isfile(proposed_label_path):
            return proposed_label_path

        for p in self._candidate_label_paths(img_path):
            if osp.isfile(p):
                return p

        # nothing worked
        raise FileNotFoundError(
            "No label found for image:\n"
            f"  image: {img_path}\n"
            f"  proposed: {proposed_label_path}\n"
            f"  tried:\n    " + "\n    ".join(self._candidate_label_paths(img_path))
        )

    @staticmethod
    def _build_name_from_template(template_name: str, frame: int) -> str:
        """
        Build a filename from a template by replacing the index portion.

        Supports:
        1) image_..._frame004401.jpg  -> replace digits after 'frame'
        2) bev_C_00001.jpg            -> replace the LAST digit-run in the stem
        3) bev_back_00001m.png        -> same as (2), preserves trailing 'm'
        4) frame007.jpg               -> same as (2)

        Replaces exactly one numeric field per filename.
        """
        base = osp.basename(template_name)

        # Prefer 'frameNNNN' token if present anywhere in the filename
        m = re.search(r"(frame)(\d+)", base)
        if m:
            pad = len(m.group(2))
            return re.sub(r"frame\d+", f"frame{frame:0{pad}d}", base, count=1)

        # Otherwise: replace last run of digits in the stem, preserving any trailing suffix (e.g. 'm')
        stem, ext = osp.splitext(base)

        m2 = re.search(r"(\d+)(\D*)$", stem)
        if not m2:
            raise ValueError(
                f"Template has no 'frameNNNN' token and no trailing digits to replace: {template_name}"
            )

        digits = m2.group(1)
        suffix = m2.group(2)  # e.g. 'm' in 'bev_back_00001m'
        pad = len(digits)

        new_stem = stem[: m2.start(1)] + f"{frame:0{pad}d}" + suffix
        return new_stem + ext

    @staticmethod
    def _ensure_required_keys(seg: Dict[str, Any], keys: List[str]) -> None:
        missing = [k for k in keys if k not in seg]
        if missing:
            raise KeyError(f"Split segment missing keys: {missing}. Segment={seg}")

    def _append_segment(self, seg: Dict[str, Any]) -> None:
        """
        Expand one JSON segment into per-frame filenames using segment templates.
        """
        self._ensure_required_keys(
            seg,
            ["start_image", "start_label", "start_frame_num", "end_frame_num"],
        )

        start = int(seg["start_frame_num"])
        end = int(seg["end_frame_num"])
        if end < start:
            raise ValueError(f"Bad segment range: start_frame_num={start} end_frame_num={end}")

        img_template = str(seg["start_image"])
        lb_template = str(seg["start_label"])

        for frame in range(start, end + 1):
            img_name = self._build_name_from_template(img_template, frame)
            lb_name = self._build_name_from_template(lb_template, frame)

            self.img_paths.append(osp.join(self.images_dir, img_name))
            self.lb_paths.append(osp.join(self.labels_dir, lb_name))

    # ----------------------------
    # Dataset API
    # ----------------------------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        impth = self.img_paths[idx]
        proposed_lbpth = self.lb_paths[idx]  # may or may not exist

        img = cv2.imread(impth, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {impth}")
        img = img[:, :, ::-1].copy()

        lbpth = self._resolve_label_path(impth, proposed_lbpth)
        lb = cv2.imread(lbpth, cv2.IMREAD_UNCHANGED)
        if lb is None:
            raise FileNotFoundError(f"Label not found or unreadable: {lbpth}")

        label = self._decode_label(lb)

        im_lb = {"im": img, "lb": label}

        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)

        # SAFETY NET: enforce {0,1} after any augmentation
        lb2 = im_lb["lb"]
        if isinstance(lb2, np.ndarray):
            if lb2.ndim == 3:
                lb2 = lb2[:, :, 0]  # should not happen, but normalize
            im_lb["lb"] = (lb2 > 0).astype(np.uint8)
        else:
            # if some transform made it a tensor already, handle later
            pass

        im_lb = self.to_tensor(im_lb)

        # SAFETY NET 2 (tensor): ensure {0,1} and correct dtype for loss
        lb_t = im_lb["lb"]
        if lb_t.max().item() > 1:
            lb_t = (lb_t > 0).to(lb_t.dtype)

        # NLLLoss wants Long targets
        lb_t = lb_t.long()

        return im_lb["im"].detach(), lb_t.unsqueeze(0).detach()

        # im_lb = {"im": img, "lb": label}

        # if self.trans_func is not None:
        #     im_lb = self.trans_func(im_lb)

        # im_lb = self.to_tensor(im_lb)
        # # after trans_func and to_tensor
        # lb_t = im_lb["lb"]  # tensor [H,W] likely
        # mx = int(lb_t.max().item())
        # mn = int(lb_t.min().item())
        # if mx > 1 or mn < 0:
        #     raise RuntimeError(
        #         f"Bad label values AFTER transforms/to_tensor: min={mn} max={mx}\n"
        #         f"image={impth}\nlabel={lbpth}"
        #     )

        # # training loop expects [1,H,W] per sample
        # return im_lb["im"].detach(), im_lb["lb"].unsqueeze(0).detach()

    # ----------------------------
    # Label decoding
    # ----------------------------

    # def _decode_label(self, lb: np.ndarray) -> np.ndarray:
    #     """
    #     Convert a loaded label image into a single-channel uint8 class-index map.

    #     Default implementation is intentionally conservative:
    #       - If HxW (single-channel): treat as binary via threshold.
    #       - If HxWx3: treat as binary "any channel > 127" (robust to BGR vs RGB).

    #     You should replace this method if your labels encode multiple classes via colors.
    #     """
    #     # HxW
    #     if lb.ndim == 2:
    #         return (lb > 127).astype(np.uint8)

    #     # HxWx3/4 -> take first 3 channels
    #     if lb.ndim == 3:
    #         lb3 = lb[:, :, :3]
    #         fg = (lb3[:, :, 0] > 127) | (lb3[:, :, 1] > 127) | (lb3[:, :, 2] > 127)
    #         return fg.astype(np.uint8)

    #     raise ValueError(f"Unsupported label shape: {lb.shape}")

    def _decode_label(self, lb: np.ndarray) -> np.ndarray:
        """
        Always return single-channel uint8 class indices in {0,1}.
        Works for:
        - grayscale 0/255
        - 3-channel jpg/png masks
        - jpeg artifacts (values between 0..255)
        """
        if lb.ndim == 3:
            lb = lb[:, :, :3]
            fg = (lb[:, :, 0] > 0) | (lb[:, :, 1] > 0) | (lb[:, :, 2] > 0)
            return fg.astype(np.uint8)

        if lb.ndim == 2:
            return (lb > 0).astype(np.uint8)

        raise ValueError(f"Unsupported label shape: {lb.shape}")
