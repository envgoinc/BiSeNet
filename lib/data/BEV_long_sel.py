
import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset
import numpy as np

class BEV_long_sel(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BEV_long_sel, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255
        self.lb_map = np.full(256, self.lb_ignore, dtype=np.uint8)
        self.lb_map[0] = 0  # obstacle
        self.lb_map[1] = 1  # water
        self.lb_map[2] = 2  # sky
        self.n_cats = 3
        self.to_tensor = T.ToTensor(
            mean=(0.4, 0.4, 0.4), # rgb
            std=(0.2, 0.2, 0.2),
        )


