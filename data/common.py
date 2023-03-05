import os
from typing import *

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, crop, pad, resize
from PIL import Image


class CommonImageDataset(Dataset):
    def __init__(
        self,
        img_paths: List[str],
        fixed_img_size: Tuple[int, int] = (1080, 1920),
        mask_available=True,
        return_labels=True,
        return_file_path=False,
        need_rotate=False,
        need_resize=False,
        need_crop=False,
        need_pad=False,
        padding_fill=0,
        padding_mode="constant",
        **args,
    ):
        super().__init__()
        self.img_paths = img_paths
        self.fixed_img_size = list(fixed_img_size)
        self.mask_available = mask_available
        self.return_labels = return_labels
        self.return_file_path = return_file_path

        self.need_rotate = need_rotate
        self.need_resize = need_resize
        self.need_crop = need_crop
        self.need_pad = need_pad

        self.padding_fill = padding_fill
        self.padding_mode = padding_mode

    def __len__(self):
        return len(self.img_paths)

    def _disect_path(self, path):
        folder, basename = os.path.split(os.path.abspath(path))
        filename, extension = os.path.splitext(basename)
        return folder, filename, extension

    def _transform(self, img):
        C, H, W = img.shape
        if self.need_rotate:
            if H > W:
                img = img.permute(0, 2, 1)
        if self.need_resize:
            img = resize(img, self.fixed_img_size)
        if self.need_crop:
            img = crop(img, 0, 0, *self.fixed_img_size)
        if self.need_pad:
            pad_right = abs(self.fixed_img_size[0] - W)
            pad_bottom = abs(self.fixed_img_size[1] - H)
            img = pad(img, [0, 0, pad_right, pad_bottom], self.padding_fill, self.padding_mode)
        return img

    def _get_input(self, path):
        try:
            img = to_tensor(Image.open(path, mode="r")) * 255
            img = img.float()[0:3]
            img = self._transform(img)
        except:
            print(f"can't open {path}")
            img = torch.zeros((3, 1080, 1920), dtype=torch.float32)
        return img

    def _get_label(self, filename):
        if "orig" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        try:
            mask = to_tensor(Image.open(f"{folder}/{filename}.mask", mode="r"))
            if len(mask.shape) < 3:
                mask = mask.unsqueeze(0)
            mask = self._transform(mask).squeeze().int()
        except:
            print(f"Cannot open mask file at {f'{folder}/{filename}.mask'}")
            mask = torch.zeros((1080, 1920), dtype=torch.uint8)
        return mask

    def __getitem__(self, index):
        path = self.img_paths[index]
        folder, filename, extension = self._disect_path(path)
        img = self._get_input(path)
        label = self._get_label(path)

        if not (self.return_labels or self.mask_available):
            return img

        batch = [img]
        if self.return_labels:
            batch.append(torch.tensor(label))
        if self.mask_available:
            if label:
                mask = self._get_mask(folder, filename)
            else:
                mask = torch.zeros((1080, 1920), dtype=torch.uint8)
            batch.append(mask)
        if self.return_file_path:
            batch.append(path)

        return batch
