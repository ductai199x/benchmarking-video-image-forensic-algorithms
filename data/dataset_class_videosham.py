from typing import *

from .common import *


class VideoShamAdobeDataset(CommonImageDataset):
    def __init__(
        self,
        dataset_samples: List[str],
        fixed_img_size: Tuple[int, int] = (1080, 1920),
        mask_available=True,
        return_labels=True,
        return_file_path=False,
        need_rotate=True,
        need_resize=False,
        need_crop=True,
        **args,
    ):
        super().__init__(
            dataset_samples,
            fixed_img_size,
            mask_available,
            return_labels,
            return_file_path,
            need_rotate,
            need_resize,
            need_crop,
            **args,
        )
