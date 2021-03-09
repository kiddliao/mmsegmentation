import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
import os
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .pipelines import Compose


@DATASETS.register_module()
class DeepGlobeRoadDataset(CustomDataset):
    """DeepGlobe Road dataset.

    Args:
        split (str): Split txt file for Pascal DeepGlobe.
    """

    CLASSES = ('background', 'road')

    PALETTE = [[0, 0, 0], [128, 255, 255]] #黑色和青色
    def __init__(self, split, **kwargs):
        super(DeepGlobeRoadDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
