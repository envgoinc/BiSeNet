import random
import math

import numpy as np
import cv2
import torch



class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[::-1, :, :].copy(),
            lb=lb[::-1, :].copy(),
        )


class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)
        return dict(im=im, lb=lb,)

    def adj_saturation(self, im, rate):
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape)/3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]


class RandomHue(object):

    def __init__(self, hue=0.1):
        self.hue = hue

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int32)
        shift = int(np.random.uniform(-self.hue * 180, self.hue * 180))
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        hsv = hsv.astype(np.uint8)
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return dict(im=im, lb=lb)


class RandomKeystoneWarp(object):
    """
    Simulates keystone distortion by randomly perturbing each corner within
    a box of +/- `jitter` pixels, then computing a perspective transform that
    maps those 4 perturbed corners back to a rectangle of `out_size` (W, H).

    Must be the FIRST transform in the Compose list.
    Applied with probability `p`.
    """

    def __init__(self, jitter=200, out_size=(1184, 896), p=0.2):
        self.jitter = jitter
        self.out_w, self.out_h = out_size
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() > self.p:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        h, w = im.shape[:2]

        def rand_pt(cx, cy):
            x = cx + np.random.randint(-self.jitter, self.jitter + 1)
            y = cy + np.random.randint(-self.jitter, self.jitter + 1)
            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))
            return [x, y]

        src = np.float32([
            rand_pt(0,     0    ),   # top-left
            rand_pt(w - 1, 0    ),   # top-right
            rand_pt(w - 1, h - 1),   # bottom-right
            rand_pt(0,     h - 1),   # bottom-left
        ])

        dst = np.float32([
            [0,              0             ],   # top-left
            [self.out_w - 1, 0             ],   # top-right
            [self.out_w - 1, self.out_h - 1],   # bottom-right
            [0,              self.out_h - 1],   # bottom-left
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        im_warped = cv2.warpPerspective(
            im, M, (self.out_w, self.out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        lb_warped = cv2.warpPerspective(
            lb, M, (self.out_w, self.out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return dict(im=im_warped, lb=lb_warped)


class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)


class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            RandomKeystoneWarp(jitter=200, out_size=(1184, 896), p=0.2),
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
            RandomHue(hue=0.1),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)


if __name__ == '__main__':
    pass