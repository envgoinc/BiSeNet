# # # lib/data/multi_source_dataset.py
# # import os
# # import os.path as osp
# # import json
# # import re
# # from typing import List, Tuple, Dict, Any, Optional

# # import cv2
# # import numpy as np
# # from torch.utils.data import Dataset

# # import lib.data.transform_cv2 as T


# # def _require_cfg(cfg, *attrs):
# #     missing = [a for a in attrs if not hasattr(cfg, a)]
# #     if missing:
# #         raise AttributeError(
# #             f"\n\n{'='*60}\n"
# #             f"MultiSourceJsonDataset: Missing required config attributes:\n"
# #             + "\n".join(f"  - cfg.{a}" for a in missing)
# #             + f"\n\nAdd these to your config before constructing the dataset.\n"
# #             f"{'='*60}\n"
# #         )


# # def _validate_color_map(color_map: Any, n_cats: int) -> None:
# #     if not isinstance(color_map, (list, tuple)):
# #         raise TypeError(
# #             f"\n\n{'='*60}\n"
# #             f"cfg.label_color_map must be a list/tuple of [R,G,B] lists.\n"
# #             f"Got: {type(color_map)}\n"
# #             f"{'='*60}\n"
# #         )
# #     if len(color_map) != n_cats:
# #         raise ValueError(
# #             f"\n\n{'='*60}\n"
# #             f"cfg.label_color_map has {len(color_map)} entries "
# #             f"but cfg.n_cats={n_cats}.\n"
# #             f"label_color_map must have exactly one [R,G,B] entry per class.\n"
# #             f"{'='*60}\n"
# #         )
# #     for i, entry in enumerate(color_map):
# #         if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
# #             raise ValueError(
# #                 f"\n\n{'='*60}\n"
# #                 f"cfg.label_color_map[{i}] must be a 3-element [R, G, B] list.\n"
# #                 f"Got: {entry!r}\n"
# #                 f"{'='*60}\n"
# #             )
# #         if not all(isinstance(v, int) and 0 <= v <= 255 for v in entry):
# #             raise ValueError(
# #                 f"\n\n{'='*60}\n"
# #                 f"cfg.label_color_map[{i}]={entry!r} contains invalid values.\n"
# #                 f"Each channel must be an integer in [0, 255].\n"
# #                 f"{'='*60}\n"
# #             )
# #     tuples = [tuple(c) for c in color_map]
# #     seen = set()
# #     for i, t in enumerate(tuples):
# #         if t in seen:
# #             raise ValueError(
# #                 f"\n\n{'='*60}\n"
# #                 f"cfg.label_color_map contains duplicate color {list(t)} at index {i}.\n"
# #                 f"Each class must map to a unique RGB color.\n"
# #                 f"{'='*60}\n"
# #             )
# #         seen.add(t)


# # class MultiSourceJsonDataset(Dataset):
# #     def __init__(
# #         self,
# #         cfg,
# #         images_dir: str,
# #         labels_dir: str,
# #         annpath: str,
# #         trans_func=None,
# #         mode: str = "train",
# #     ):
# #         assert mode in ("train", "val", "test"), f"mode must be train/val/test, got {mode}"
# #         self.mode = mode
# #         self.trans_func = trans_func

# #         # ---- Validate all required cfg fields up front ----
# #         _require_cfg(cfg, "n_cats", "lb_ignore", "mean", "std", "label_color_map")

# #         self.n_cats    = cfg.n_cats
# #         self.lb_ignore = cfg.lb_ignore

# #         _validate_color_map(cfg.label_color_map, self.n_cats)

# #         # Build fast lookup: (R, G, B) -> class_index
# #         self._color_to_class: Dict[Tuple[int, int, int], int] = {
# #             tuple(rgb): idx for idx, rgb in enumerate(cfg.label_color_map)
# #         }

# #         self.to_tensor = T.ToTensor(mean=tuple(cfg.mean), std=tuple(cfg.std))

# #         self.images_dir = osp.abspath(images_dir)
# #         self.labels_dir = osp.abspath(labels_dir)
# #         annpath         = osp.abspath(annpath)

# #         if not osp.isdir(self.images_dir):
# #             raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
# #         if not osp.isdir(self.labels_dir):
# #             raise FileNotFoundError(f"labels_dir not found: {self.labels_dir}")
# #         if not osp.isfile(annpath):
# #             raise FileNotFoundError(f"annpath not found: {annpath}")

# #         with open(annpath, "r") as f:
# #             spec = json.load(f)

# #         if "splits" not in spec or mode not in spec["splits"]:
# #             raise KeyError(f"JSON spec missing splits['{mode}']")

# #         segments = spec["splits"][mode]
# #         if not isinstance(segments, list) or len(segments) == 0:
# #             raise ValueError(f"splits['{mode}'] is empty or not a list")

# #         self.img_paths = []
# #         self.lb_paths  = []
# #         for seg in segments:
# #             self._append_segment(seg)

# #         if len(self.img_paths) != len(self.lb_paths):
# #             raise RuntimeError("img_paths and lb_paths length mismatch")
# #         if len(self.img_paths) == 0:
# #             raise RuntimeError("No samples found after expanding splits")

# #         self._len = len(self.img_paths)

# #     # ----------------------------
# #     # Path resolution / expansion
# #     # ----------------------------

# #     def _candidate_label_paths(self, img_path: str) -> list[str]:
# #         img_base = osp.basename(img_path)
# #         cands    = [osp.join(self.labels_dir, img_base)]
# #         stem, ext = osp.splitext(img_base)
# #         cands.append(osp.join(self.labels_dir, f"{stem}m{ext}"))
# #         cands.append(osp.join(self.labels_dir, f"{stem}m.png"))
# #         if stem.startswith("image_"):
# #             cands.append(osp.join(self.labels_dir, f"label_{stem[len('image_'):]}{ext}"))
# #             cands.append(osp.join(self.labels_dir, f"label_{stem[len('image_'):]}.png"))
# #         return cands

# #     def _resolve_label_path(self, img_path: str, proposed_label_path: str | None) -> str:
# #         if proposed_label_path and osp.isfile(proposed_label_path):
# #             return proposed_label_path
# #         for p in self._candidate_label_paths(img_path):
# #             if osp.isfile(p):
# #                 return p
# #         raise FileNotFoundError(
# #             "No label found for image:\n"
# #             f"  image:    {img_path}\n"
# #             f"  proposed: {proposed_label_path}\n"
# #             f"  tried:\n    " + "\n    ".join(self._candidate_label_paths(img_path))
# #         )

# #     @staticmethod
# #     def _build_name_from_template(template_name: str, frame: int) -> str:
# #         base = osp.basename(template_name)
# #         m = re.search(r"(frame)(\d+)", base)
# #         if m:
# #             pad = len(m.group(2))
# #             return re.sub(r"frame\d+", f"frame{frame:0{pad}d}", base, count=1)
# #         stem, ext = osp.splitext(base)
# #         m2 = re.search(r"(\d+)(\D*)$", stem)
# #         if not m2:
# #             raise ValueError(
# #                 f"Template has no 'frameNNNN' token and no trailing digits to replace: {template_name}"
# #             )
# #         digits = m2.group(1)
# #         suffix = m2.group(2)
# #         pad    = len(digits)
# #         new_stem = stem[: m2.start(1)] + f"{frame:0{pad}d}" + suffix
# #         return new_stem + ext

# #     @staticmethod
# #     def _ensure_required_keys(seg: Dict[str, Any], keys: List[str]) -> None:
# #         missing = [k for k in keys if k not in seg]
# #         if missing:
# #             raise KeyError(f"Split segment missing keys: {missing}. Segment={seg}")

# #     def _append_segment(self, seg: Dict[str, Any]) -> None:
# #         self._ensure_required_keys(
# #             seg,
# #             ["start_image", "start_label", "start_frame_num", "end_frame_num"],
# #         )
# #         start = int(seg["start_frame_num"])
# #         end   = int(seg["end_frame_num"])
# #         if end < start:
# #             raise ValueError(f"Bad segment range: start_frame_num={start} end_frame_num={end}")
# #         img_template = str(seg["start_image"])
# #         lb_template  = str(seg["start_label"])
# #         for frame in range(start, end + 1):
# #             img_name = self._build_name_from_template(img_template, frame)
# #             lb_name  = self._build_name_from_template(lb_template, frame)
# #             self.img_paths.append(osp.join(self.images_dir, img_name))
# #             self.lb_paths.append(osp.join(self.labels_dir, lb_name))

# #     # ----------------------------
# #     # Dataset API
# #     # ----------------------------

# #     def __len__(self) -> int:
# #         return self._len

# #     def __getitem__(self, idx: int):
# #         impth          = self.img_paths[idx]
# #         proposed_lbpth = self.lb_paths[idx]

# #         img = cv2.imread(impth, cv2.IMREAD_COLOR)
# #         if img is None:
# #             raise FileNotFoundError(f"Image not found or unreadable: {impth}")
# #         img = img[:, :, ::-1].copy()

# #         lbpth = self._resolve_label_path(impth, proposed_lbpth)
# #         lb    = cv2.imread(lbpth, cv2.IMREAD_UNCHANGED)
# #         if lb is None:
# #             raise FileNotFoundError(f"Label not found or unreadable: {lbpth}")

# #         label = self._decode_label(lb, lbpth)

# #         im_lb = {"im": img, "lb": label}

# #         if self.trans_func is not None:
# #             im_lb = self.trans_func(im_lb)

# #         im_lb = self.to_tensor(im_lb)
# #         lb_t  = im_lb["lb"].long()

# #         return im_lb["im"].detach(), lb_t.unsqueeze(0).detach()

# #     # ----------------------------
# #     # Label decoding
# #     # ----------------------------

# #     def _decode_label(self, lb: np.ndarray, lbpth: str = "<unknown>") -> np.ndarray:
# #         if lb.ndim == 3 and lb.shape[2] == 4:
# #             lb = lb[:, :, :3]

# #         if lb.ndim == 3:
# #             lb_rgb = lb[:, :, ::-1]  # cv2 BGR -> RGB
# #         elif lb.ndim == 2:
# #             lb_rgb = np.stack([lb, lb, lb], axis=-1)
# #         else:
# #             raise ValueError(f"Unsupported label shape {lb.shape} in: {lbpth}")

# #         H, W = lb_rgb.shape[:2]
# #         out  = np.full((H, W), self.lb_ignore, dtype=np.uint8)

# #         for (r, g, b), class_idx in self._color_to_class.items():
# #             mask       = (lb_rgb[:, :, 0] == r) & (lb_rgb[:, :, 1] == g) & (lb_rgb[:, :, 2] == b)
# #             out[mask]  = class_idx

# #         ignored     = int((out == self.lb_ignore).sum())
# #         ignore_frac = ignored / (H * W)
# #         if ignore_frac > 0.10:
# #             import warnings
# #             warnings.warn(
# #                 f"\n{'='*60}\n"
# #                 f"_decode_label: {ignore_frac:.1%} of pixels are UNMAPPED "
# #                 f"(assigned lb_ignore={self.lb_ignore}).\n"
# #                 f"  label file : {lbpth}\n"
# #                 f"  label shape: {lb.shape}\n"
# #                 f"  Expected colors (RGB): {list(self._color_to_class.keys())}\n"
# #                 f"  Unique pixel values found (first 10): "
# #                 f"{[tuple(x) for x in np.unique(lb_rgb.reshape(-1, 3), axis=0)[:10].tolist()]}\n"
# #                 f"Check cfg.label_color_map and confirm masks are saved as PNG (not JPEG).\n"
# #                 f"{'='*60}",
# #                 stacklevel=2,
# #             )

# #         return out


# # lib/data/multi_source_dataset.py
# import os
# import os.path as osp
# import json
# import re
# from typing import List, Tuple, Dict, Any, Optional

# import cv2
# import numpy as np
# from torch.utils.data import Dataset

# import lib.data.transform_cv2 as T


# def _require_cfg(cfg, *attrs):
#     missing = [a for a in attrs if not hasattr(cfg, a)]
#     if missing:
#         raise AttributeError(
#             f"\n\n{'='*60}\n"
#             f"MultiSourceJsonDataset: Missing required config attributes:\n"
#             + "\n".join(f"  - cfg.{a}" for a in missing)
#             + f"\n\nAdd these to your config before constructing the dataset.\n"
#             f"{'='*60}\n"
#         )


# def _validate_color_map(color_map: Any, n_cats: int) -> None:
#     if not isinstance(color_map, (list, tuple)):
#         raise TypeError(
#             f"\n\n{'='*60}\n"
#             f"cfg.label_color_map must be a list/tuple of [R,G,B] lists.\n"
#             f"Got: {type(color_map)}\n"
#             f"{'='*60}\n"
#         )
#     if len(color_map) != n_cats:
#         raise ValueError(
#             f"\n\n{'='*60}\n"
#             f"cfg.label_color_map has {len(color_map)} entries "
#             f"but cfg.n_cats={n_cats}.\n"
#             f"label_color_map must have exactly one [R,G,B] entry per class.\n"
#             f"{'='*60}\n"
#         )
#     for i, entry in enumerate(color_map):
#         if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
#             raise ValueError(
#                 f"\n\n{'='*60}\n"
#                 f"cfg.label_color_map[{i}] must be a 3-element [R, G, B] list.\n"
#                 f"Got: {entry!r}\n"
#                 f"{'='*60}\n"
#             )
#         if not all(isinstance(v, int) and 0 <= v <= 255 for v in entry):
#             raise ValueError(
#                 f"\n\n{'='*60}\n"
#                 f"cfg.label_color_map[{i}]={entry!r} contains invalid values.\n"
#                 f"Each channel must be an integer in [0, 255].\n"
#                 f"{'='*60}\n"
#             )
#     tuples = [tuple(c) for c in color_map]
#     seen = set()
#     for i, t in enumerate(tuples):
#         if t in seen:
#             raise ValueError(
#                 f"\n\n{'='*60}\n"
#                 f"cfg.label_color_map contains duplicate color {list(t)} at index {i}.\n"
#                 f"Each class must map to a unique RGB color.\n"
#                 f"{'='*60}\n"
#             )
#         seen.add(t)


# class MultiSourceJsonDataset(Dataset):
#     def __init__(
#         self,
#         cfg,
#         images_dir: str,
#         labels_dir: str,
#         annpath: str,
#         trans_func=None,
#         mode: str = "train",
#     ):
#         assert mode in ("train", "val", "test"), f"mode must be train/val/test, got {mode}"
#         self.mode      = mode
#         self.trans_func = trans_func

#         _require_cfg(cfg, "n_cats", "lb_ignore", "mean", "std", "label_color_map")

#         self.n_cats    = cfg.n_cats
#         self.lb_ignore = cfg.lb_ignore

#         _validate_color_map(cfg.label_color_map, self.n_cats)

#         # (R,G,B) -> class_index, kept for warning messages only
#         self._color_to_class: Dict[Tuple[int, int, int], int] = {
#             tuple(rgb): idx for idx, rgb in enumerate(cfg.label_color_map)
#         }

#         # --- Fast decode: 16M-entry uint8 LUT indexed by packed RGB uint32 ---
#         # Built once at init, zero-copy lookup at decode time.
#         # All entries default to lb_ignore; only known colors get their class index.
#         lut = np.full(1 << 24, self.lb_ignore, dtype=np.uint8)
#         for (r, g, b), idx in self._color_to_class.items():
#             lut[r << 16 | g << 8 | b] = idx
#         self._lut = lut  # shape (16777216,) uint8

#         self.to_tensor = T.ToTensor(mean=tuple(cfg.mean), std=tuple(cfg.std))

#         self.images_dir = osp.abspath(images_dir)
#         self.labels_dir = osp.abspath(labels_dir)
#         annpath         = osp.abspath(annpath)

#         if not osp.isdir(self.images_dir):
#             raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
#         if not osp.isdir(self.labels_dir):
#             raise FileNotFoundError(f"labels_dir not found: {self.labels_dir}")
#         if not osp.isfile(annpath):
#             raise FileNotFoundError(f"annpath not found: {annpath}")

#         with open(annpath, "r") as f:
#             spec = json.load(f)

#         if "splits" not in spec or mode not in spec["splits"]:
#             raise KeyError(f"JSON spec missing splits['{mode}']")

#         segments = spec["splits"][mode]
#         if not isinstance(segments, list) or len(segments) == 0:
#             raise ValueError(f"splits['{mode}'] is empty or not a list")

#         self.img_paths = []
#         self.lb_paths  = []
#         for seg in segments:
#             self._append_segment(seg)

#         if len(self.img_paths) != len(self.lb_paths):
#             raise RuntimeError("img_paths and lb_paths length mismatch")
#         if len(self.img_paths) == 0:
#             raise RuntimeError("No samples found after expanding splits")

#         self._len = len(self.img_paths)

#     # ----------------------------
#     # Path resolution / expansion
#     # ----------------------------

#     def _candidate_label_paths(self, img_path: str) -> list[str]:
#         img_base  = osp.basename(img_path)
#         stem, ext = osp.splitext(img_base)
#         cands = [
#             osp.join(self.labels_dir, img_base),
#             osp.join(self.labels_dir, f"{stem}.png"),
#             osp.join(self.labels_dir, f"{stem}m{ext}"),
#             osp.join(self.labels_dir, f"{stem}m.png"),
#         ]
#         if stem.startswith("image_"):
#             tail = stem[len("image_"):]
#             cands.append(osp.join(self.labels_dir, f"label_{tail}{ext}"))
#             cands.append(osp.join(self.labels_dir, f"label_{tail}.png"))
#         return cands

#     def _resolve_label_path(self, img_path: str, proposed_label_path: str | None) -> str:
#         if proposed_label_path and osp.isfile(proposed_label_path):
#             return proposed_label_path
#         for p in self._candidate_label_paths(img_path):
#             if osp.isfile(p):
#                 return p
#         raise FileNotFoundError(
#             "No label found for image:\n"
#             f"  image:    {img_path}\n"
#             f"  proposed: {proposed_label_path}\n"
#             f"  tried:\n    " + "\n    ".join(self._candidate_label_paths(img_path))
#         )

#     @staticmethod
#     def _build_name_from_template(template_name: str, frame: int) -> str:
#         base = osp.basename(template_name)
#         m = re.search(r"(frame)(\d+)", base)
#         if m:
#             pad = len(m.group(2))
#             return re.sub(r"frame\d+", f"frame{frame:0{pad}d}", base, count=1)
#         stem, ext = osp.splitext(base)
#         m2 = re.search(r"(\d+)(\D*)$", stem)
#         if not m2:
#             raise ValueError(
#                 f"Template has no 'frameNNNN' token and no trailing digits to replace: {template_name}"
#             )
#         pad      = len(m2.group(1))
#         suffix   = m2.group(2)
#         new_stem = stem[: m2.start(1)] + f"{frame:0{pad}d}" + suffix
#         return new_stem + ext

#     @staticmethod
#     def _ensure_required_keys(seg: Dict[str, Any], keys: List[str]) -> None:
#         missing = [k for k in keys if k not in seg]
#         if missing:
#             raise KeyError(f"Split segment missing keys: {missing}. Segment={seg}")

#     def _append_segment(self, seg: Dict[str, Any]) -> None:
#         self._ensure_required_keys(
#             seg, ["start_image", "start_label", "start_frame_num", "end_frame_num"],
#         )
#         start = int(seg["start_frame_num"])
#         end   = int(seg["end_frame_num"])
#         if end < start:
#             raise ValueError(f"Bad segment range: start_frame_num={start} end_frame_num={end}")
#         img_template = str(seg["start_image"])
#         lb_template  = str(seg["start_label"])
#         for frame in range(start, end + 1):
#             self.img_paths.append(osp.join(self.images_dir, self._build_name_from_template(img_template, frame)))
#             self.lb_paths.append(osp.join(self.labels_dir,  self._build_name_from_template(lb_template,  frame)))

#     # ----------------------------
#     # Dataset API
#     # ----------------------------

#     def __len__(self) -> int:
#         return self._len

#     def __getitem__(self, idx: int):
#         impth          = self.img_paths[idx]
#         proposed_lbpth = self.lb_paths[idx]

#         img = cv2.imread(impth, cv2.IMREAD_COLOR)
#         if img is None:
#             raise FileNotFoundError(f"Image not found or unreadable: {impth}")
#         img = img[:, :, ::-1].copy()  # BGR -> RGB

#         lbpth = self._resolve_label_path(impth, proposed_lbpth)
#         lb    = cv2.imread(lbpth, cv2.IMREAD_UNCHANGED)
#         if lb is None:
#             raise FileNotFoundError(f"Label not found or unreadable: {lbpth}")

#         label = self._decode_label(lb, lbpth)

#         im_lb = {"im": img, "lb": label}
#         if self.trans_func is not None:
#             im_lb = self.trans_func(im_lb)

#         im_lb = self.to_tensor(im_lb)
#         lb_t  = im_lb["lb"].long()

#         return im_lb["im"].detach(), lb_t.unsqueeze(0).detach()

#     # ----------------------------
#     # Label decoding
#     # ----------------------------

#     def _decode_label(self, lb: np.ndarray, lbpth: str = "<unknown>") -> np.ndarray:
#         if lb.ndim == 3 and lb.shape[2] == 4:
#             lb = lb[:, :, :3]

#         if lb.ndim == 3:
#             # cv2 loads BGR — flip to RGB in one contiguous copy
#             lb_rgb = lb[:, :, ::-1].copy()
#         elif lb.ndim == 2:
#             lb_rgb = np.stack([lb, lb, lb], axis=-1)
#         else:
#             raise ValueError(f"Unsupported label shape {lb.shape} in: {lbpth}")

#         # Pack R,G,B -> uint32 index in one vectorized op, then index LUT
#         packed = (lb_rgb[:, :, 0].astype(np.uint32) << 16
#                 | lb_rgb[:, :, 1].astype(np.uint32) << 8
#                 |  lb_rgb[:, :, 2].astype(np.uint32))
#         out = self._lut[packed]  # single array index op, no Python loop

#         # Warning only fires when something is actually wrong — np.unique is expensive
#         # so it's only called here, never on the hot path
#         ignored = int((out == self.lb_ignore).sum())
#         if ignored / out.size > 0.10:
#             import warnings
#             unique_colors = [tuple(x) for x in np.unique(lb_rgb.reshape(-1, 3), axis=0)[:10].tolist()]
#             warnings.warn(
#                 f"\n{'='*60}\n"
#                 f"_decode_label: {ignored/out.size:.1%} of pixels are UNMAPPED.\n"
#                 f"  label file : {lbpth}\n"
#                 f"  Expected colors (RGB): {list(self._color_to_class.keys())}\n"
#                 f"  Unique pixel values found (first 10): {unique_colors}\n"
#                 f"Check cfg.label_color_map and confirm masks are saved as PNG.\n"
#                 f"{'='*60}",
#                 stacklevel=2,
#             )

#         return out


# lib/data/multi_source_dataset.py
import os
import os.path as osp
import json
import re
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

import lib.data.transform_cv2 as T


def _require_cfg(cfg, *attrs):
    missing = [a for a in attrs if not hasattr(cfg, a)]
    if missing:
        raise AttributeError(
            f"\n\n{'='*60}\n"
            f"MultiSourceJsonDataset: Missing required config attributes:\n"
            + "\n".join(f"  - cfg.{a}" for a in missing)
            + f"\n\nAdd these to your config before constructing the dataset.\n"
            f"{'='*60}\n"
        )


def _validate_color_map(color_map: Any, n_cats: int) -> None:
    if not isinstance(color_map, (list, tuple)):
        raise TypeError(
            f"\n\n{'='*60}\n"
            f"cfg.label_color_map must be a list/tuple of [R,G,B] lists.\n"
            f"Got: {type(color_map)}\n"
            f"{'='*60}\n"
        )
    if len(color_map) != n_cats:
        raise ValueError(
            f"\n\n{'='*60}\n"
            f"cfg.label_color_map has {len(color_map)} entries "
            f"but cfg.n_cats={n_cats}.\n"
            f"label_color_map must have exactly one [R,G,B] entry per class.\n"
            f"{'='*60}\n"
        )
    for i, entry in enumerate(color_map):
        if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
            raise ValueError(
                f"\n\n{'='*60}\n"
                f"cfg.label_color_map[{i}] must be a 3-element [R, G, B] list.\n"
                f"Got: {entry!r}\n"
                f"{'='*60}\n"
            )
        if not all(isinstance(v, int) and 0 <= v <= 255 for v in entry):
            raise ValueError(
                f"\n\n{'='*60}\n"
                f"cfg.label_color_map[{i}]={entry!r} contains invalid values.\n"
                f"Each channel must be an integer in [0, 255].\n"
                f"{'='*60}\n"
            )
    tuples = [tuple(c) for c in color_map]
    seen = set()
    for i, t in enumerate(tuples):
        if t in seen:
            raise ValueError(
                f"\n\n{'='*60}\n"
                f"cfg.label_color_map contains duplicate color {list(t)} at index {i}.\n"
                f"Each class must map to a unique RGB color.\n"
                f"{'='*60}\n"
            )
        seen.add(t)


class MultiSourceJsonDataset(Dataset):
    def __init__(
        self,
        cfg,
        images_dir: str,
        labels_dir: str,
        annpath: str,
        trans_func=None,
        mode: str = "train",
    ):
        assert mode in ("train", "val", "test"), f"mode must be train/val/test, got {mode}"
        self.mode       = mode
        self.trans_func = trans_func

        _require_cfg(cfg, "n_cats", "lb_ignore", "mean", "std", "label_color_map")

        self.n_cats    = cfg.n_cats
        self.lb_ignore = cfg.lb_ignore

        _validate_color_map(cfg.label_color_map, self.n_cats)

        # (R,G,B) -> class_index, kept for warning messages only
        self._color_to_class: Dict[Tuple[int, int, int], int] = {
            tuple(rgb): idx for idx, rgb in enumerate(cfg.label_color_map)
        }

        # 16M-entry uint8 LUT indexed by packed RGB uint32, built once at init
        lut = np.full(1 << 24, self.lb_ignore, dtype=np.uint8)
        for (r, g, b), idx in self._color_to_class.items():
            lut[r << 16 | g << 8 | b] = idx
        self._lut = lut

        self.to_tensor = T.ToTensor(mean=tuple(cfg.mean), std=tuple(cfg.std))

        self.images_dir = osp.abspath(images_dir)
        self.labels_dir = osp.abspath(labels_dir)
        annpath         = osp.abspath(annpath)

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
        self.lb_paths  = []
        for seg in segments:
            self._append_segment(seg)

        if len(self.img_paths) != len(self.lb_paths):
            raise RuntimeError("img_paths and lb_paths length mismatch")
        if len(self.img_paths) == 0:
            raise RuntimeError("No samples found after expanding splits")

        # Resolve all label paths once at init — zero filesystem calls in __getitem__
        print(f"[MultiSourceJsonDataset] Resolving {len(self.lb_paths)} label paths ({mode})...")
        resolved = []
        missing  = []
        for img_path, lb_path in zip(self.img_paths, self.lb_paths):
            try:
                resolved.append(self._resolve_label_path(img_path, lb_path))
            except FileNotFoundError:
                missing.append((img_path, lb_path))

        if missing:
            lines = "\n".join(f"  img={i}\n  lbl={l}" for i, l in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} images have no matching label. First 5:\n{lines}"
            )

        self.lb_paths = resolved  # fully resolved, verified — safe to use directly
        self._len     = len(self.img_paths)

    # ----------------------------
    # Path resolution / expansion
    # ----------------------------

    def _candidate_label_paths(self, img_path: str) -> list[str]:
        img_base  = osp.basename(img_path)
        stem, ext = osp.splitext(img_base)
        cands = [
            osp.join(self.labels_dir, img_base),
            osp.join(self.labels_dir, f"{stem}.png"),
            osp.join(self.labels_dir, f"{stem}m{ext}"),
            osp.join(self.labels_dir, f"{stem}m.png"),
        ]
        if stem.startswith("image_"):
            tail = stem[len("image_"):]
            cands.append(osp.join(self.labels_dir, f"label_{tail}{ext}"))
            cands.append(osp.join(self.labels_dir, f"label_{tail}.png"))
        return cands

    def _resolve_label_path(self, img_path: str, proposed_label_path: str | None) -> str:
        if proposed_label_path and osp.isfile(proposed_label_path):
            return proposed_label_path
        for p in self._candidate_label_paths(img_path):
            if osp.isfile(p):
                return p
        raise FileNotFoundError(
            "No label found for image:\n"
            f"  image:    {img_path}\n"
            f"  proposed: {proposed_label_path}\n"
            f"  tried:\n    " + "\n    ".join(self._candidate_label_paths(img_path))
        )

    @staticmethod
    def _build_name_from_template(template_name: str, frame: int) -> str:
        base = osp.basename(template_name)
        m = re.search(r"(frame)(\d+)", base)
        if m:
            pad = len(m.group(2))
            return re.sub(r"frame\d+", f"frame{frame:0{pad}d}", base, count=1)
        stem, ext = osp.splitext(base)
        m2 = re.search(r"(\d+)(\D*)$", stem)
        if not m2:
            raise ValueError(
                f"Template has no 'frameNNNN' token and no trailing digits to replace: {template_name}"
            )
        pad      = len(m2.group(1))
        suffix   = m2.group(2)
        new_stem = stem[: m2.start(1)] + f"{frame:0{pad}d}" + suffix
        return new_stem + ext

    @staticmethod
    def _ensure_required_keys(seg: Dict[str, Any], keys: List[str]) -> None:
        missing = [k for k in keys if k not in seg]
        if missing:
            raise KeyError(f"Split segment missing keys: {missing}. Segment={seg}")

    def _append_segment(self, seg: Dict[str, Any]) -> None:
        self._ensure_required_keys(
            seg, ["start_image", "start_label", "start_frame_num", "end_frame_num"],
        )
        start = int(seg["start_frame_num"])
        end   = int(seg["end_frame_num"])
        if end < start:
            raise ValueError(f"Bad segment range: start_frame_num={start} end_frame_num={end}")
        img_template = str(seg["start_image"])
        lb_template  = str(seg["start_label"])
        for frame in range(start, end + 1):
            self.img_paths.append(osp.join(self.images_dir, self._build_name_from_template(img_template, frame)))
            self.lb_paths.append(osp.join(self.labels_dir,  self._build_name_from_template(lb_template,  frame)))

    # ----------------------------
    # Dataset API
    # ----------------------------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        impth = self.img_paths[idx]
        lbpth = self.lb_paths[idx]  # pre-resolved at init, guaranteed to exist

        img = cv2.imread(impth, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {impth}")
        img = img[:, :, ::-1].copy()

        lb = cv2.imread(lbpth, cv2.IMREAD_UNCHANGED)
        if lb is None:
            raise FileNotFoundError(f"Label not found or unreadable: {lbpth}")

        label = self._decode_label(lb, lbpth)

        im_lb = {"im": img, "lb": label}
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)

        im_lb = self.to_tensor(im_lb)
        lb_t  = im_lb["lb"].long()

        return im_lb["im"].detach(), lb_t.unsqueeze(0).detach()

    # ----------------------------
    # Label decoding
    # ----------------------------

    def _decode_label(self, lb: np.ndarray, lbpth: str = "<unknown>") -> np.ndarray:
        if lb.ndim == 3 and lb.shape[2] == 4:
            lb = lb[:, :, :3]

        if lb.ndim == 3:
            lb_rgb = lb[:, :, ::-1].copy()  # BGR -> RGB, contiguous
        elif lb.ndim == 2:
            lb_rgb = np.stack([lb, lb, lb], axis=-1)
        else:
            raise ValueError(f"Unsupported label shape {lb.shape} in: {lbpth}")

        packed = (lb_rgb[:, :, 0].astype(np.uint32) << 16
                | lb_rgb[:, :, 1].astype(np.uint32) << 8
                | lb_rgb[:, :, 2].astype(np.uint32))
        out = self._lut[packed]

        ignored = int((out == self.lb_ignore).sum())
        if ignored / out.size > 0.10:
            import warnings
            unique_colors = [tuple(x) for x in np.unique(lb_rgb.reshape(-1, 3), axis=0)[:10].tolist()]
            warnings.warn(
                f"\n{'='*60}\n"
                f"_decode_label: {ignored/out.size:.1%} of pixels are UNMAPPED.\n"
                f"  label file : {lbpth}\n"
                f"  Expected colors (RGB): {list(self._color_to_class.keys())}\n"
                f"  Unique pixel values found (first 10): {unique_colors}\n"
                f"Check cfg.label_color_map and confirm masks are saved as PNG.\n"
                f"{'='*60}",
                stacklevel=2,
            )

        return out