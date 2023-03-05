from data.common import *


class FaceForensicsPlusPlusDataset(CommonImageDataset):
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

    def _get_mask(self, folder: str, filename: str):
        mask_path = f"{folder}/{filename.replace('manip', 'mask')}.png"
        try:
            mask = to_tensor(Image.open(mask_path, mode="r")).sum(dim=0)
            if len(mask.shape) < 3:
                mask = mask.unsqueeze(0)
            mask[mask > 0] = 1
            mask = self._transform(mask).squeeze().int()
        except:
            print(f"Cannot open mask file at {mask_path}")
            mask = torch.zeros((1080, 1920), dtype=torch.uint8)
        return mask
