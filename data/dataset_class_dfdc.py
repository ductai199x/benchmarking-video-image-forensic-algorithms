from typing import *

from .common import *


class DFDCDataset(CommonImageDataset):
    def __init__(
        self,
        dataset_samples: List[str],
        fixed_img_size: Tuple[int, int] = (1080, 1920),
        mask_available=True,
        return_labels=True,
        return_file_path=False,
        need_rotate=True,
        need_resize=True,
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

    def _get_mask(self, folder, filename):
        # try:
        #     mask = to_tensor(Image.open(f"{folder}/{filename}.mask", mode="r"))
        #     if len(mask.shape) < 3:
        #         mask = mask.unsqueeze(0)
        #     mask = self._transform(mask).squeeze().int()
        # except:
        #     print(f"Cannot open mask file at {f'{folder}/{filename}.mask'}")
        #     mask = torch.zeros((1080, 1920), dtype=torch.uint8)
        mask = torch.zeros((1080, 1920), dtype=torch.uint8)
        return mask
