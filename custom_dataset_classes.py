import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import crop, pad, resize
from torchvision.transforms import ToTensor
from PIL import Image

from helper import get_all_files


class GenericImageDataset(Dataset):
    def __init__(self, img_paths, mask_available=True, return_labels=True):
        self.img_paths = img_paths
        self.mask_available = mask_available
        self.return_labels = return_labels
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def _disect_path(self, path):
        folder, basename = os.path.split(os.path.abspath(path))
        filename, extension = os.path.splitext(basename)
        return folder, filename, extension

    def _get_input(self, path):
        try:
            img = Image.open(path, mode="r")
            img = self.to_tensor(img) * 255
            img = img.float()[0:3]
            return img
        except:
            print(f"can't open {path}")
            return torch.zeros(3, 1080, 1920)

    def _get_label(self, filename):
        if "orig" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        mask = self.to_tensor(
            Image.open(f"{folder}/{filename}.mask", mode="r")
        ).squeeze()
        return mask.int()

    def __getitem__(self, index):
        path = self.img_paths[index]
        folder, filename, extension = self._disect_path(path)
        img = self._get_input(path)
        label = self._get_label(path)

        if not (self.return_labels or self.mask_available):
            return img

        batch = [img]
        if self.return_labels:
            batch.append(label)
        if self.mask_available:
            if label:
                mask = self._get_mask(folder, filename)
            else:
                mask = torch.zeros(1080, 1920).to(torch.uint8)
            batch.append(mask)

        return batch


class CarvalhoImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = Image.open(path, mode="r")
        img = self.to_tensor(img) * 255
        img = img[0:3]
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        if img.shape[1] != 1080:
            img = crop(img, 0, 0, 1080, 1920)
        return img.float()

    def _get_label(self, filename):
        if "normal" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("carvalho_tampered", "carvalho_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}_", suffix=".png")[0]
        mask = read_image(path)
        mask = 1 - (mask.sum(dim=0) / (255.0 * mask.shape[0]))
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask


class KorusImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = Image.open(path, mode="r")
        img = self.to_tensor(img) * 255
        img = img[0:3]
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        if img.shape[1] != 1080:
            img = crop(img, 0, 0, 1080, 1920)
        return img.float()

    def _get_label(self, filename):
        if "normal" in filename:
            return 0
        else:
            return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("korus_tampered", "korus_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}.", suffix=".PNG")[0]
        mask = read_image(path)
        mask = mask.sum(dim=0) / (255.0 * mask.shape[0])
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask


# Huh's In The Wild dataset -- only edited images
class ITWImageDataset(GenericImageDataset):
    def _get_input(self, path):
        img = Image.open(path, mode="r")
        img = self.to_tensor(img) * 255
        if img.shape[1] > img.shape[2]:
            img = img.permute(0, 2, 1)
        if img.shape[1] != 1080:
            print(img.shape)
            img = crop(img, 0, 0, 1080, 1920)

        return img

    def _get_label(self, filename):
        return 1

    def _get_mask(self, folder, filename):
        folder = folder.replace("huh_in_the_wild_tampered", "huh_in_the_wild_masks")
        path = get_all_files(f"{folder}", prefix=f"{filename}.", suffix=".png")[0]
        mask = read_image(path)
        mask = mask.sum(dim=0) / (255.0 * mask.shape[0])
        if mask.shape[0] > mask.shape[1]:
            mask = mask.permute(1, 0)
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask


class VideoShamAdobeDataset(GenericImageDataset):
    def _get_input(self, path):
        try:
            img = Image.open(path, mode="r")
            img = self.to_tensor(img) * 255
            img = img.float()[0:3]
            if img.shape[1] != 1080:
                img = crop(img, 0, 0, 1080, 1920)
            return img
        except:
            print(f"can't open {path}")
            return torch.zeros(3, 1080, 1920)

    def _get_mask(self, folder, filename):
        mask = self.to_tensor(
            Image.open(f"{folder}/{filename}.mask", mode="r")
        ).squeeze()
        if mask.shape[0] != 1080:
            mask = crop(mask, 0, 0, 1080, 1920)
        return mask.to(torch.uint8)

    def __getitem__(self, index):
        path = self.img_paths[index]
        folder, filename, extension = self._disect_path(path)
        img = self._get_input(path)

        label = self._get_label(path)

        if not (self.return_labels or self.mask_available):
            return img

        batch = [img]
        if self.return_labels:
            batch.append(label)
        if self.mask_available:
            if label:
                mask = self._get_mask(folder, filename)
            else:
                mask = torch.zeros(1080, 1920).to(torch.uint8)
            batch.append(mask)

        return batch


class E2fgviDavisDataset(GenericImageDataset):
    def __init__(
        self,
        img_paths,
        resolution=(1080, 1920),
        mask_available=True,
        return_labels=True,
    ):
        super().__init__(img_paths, mask_available, return_labels)
        self.resolution = resolution

    def _get_input(self, path):
        try:
            img = Image.open(path, mode="r")
            img = self.to_tensor(img) * 255
            img = img.float()[0:3]
            if img.shape[1] != self.resolution[0]:
                img = resize(img, self.resolution)
            return img
        except:
            print(f"can't open {path}")
            return torch.zeros(3, self.resolution[0], self.resolution[1])

    def _get_mask(self, folder, filename):
        mask = self.to_tensor(Image.open(f"{folder}/{filename}.mask", mode="r"))
        if mask.shape[1] != self.resolution[0]:
            mask = resize(mask, self.resolution)

        mask = mask.squeeze(0)
        mask[mask > 0] = 1

        return mask.to(torch.uint8)
