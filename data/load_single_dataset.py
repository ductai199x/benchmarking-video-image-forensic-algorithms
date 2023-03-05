import sys

sys.path.append("..")

import random
from typing import Literal

from torch.utils.data import DataLoader

from utils import get_all_files, list_dir

from .common import CommonImageDataset
from .dataset_class_dfd import DeepfakeDetectionDataset
from .dataset_class_dfdc import DFDCDataset
from .dataset_class_ffpp import FaceForensicsPlusPlusDataset
from .dataset_class_inpaint import InpaintingDataset
from .dataset_class_videosham import VideoShamAdobeDataset
from .dataset_paths import DATASET_ROOT_PATH


def load_single_dataset(
    dataset_name,
    dataset_split: Literal["train", "val", "test"] = "test",
    shuffle=False,
    num_samples=-1,
    filepath_contains="",
    batch_size=10,
    num_workers=10,
    **args,
):
    random.seed(42)
    if dataset_name in ["vcms", "vpvm", "vpim", "icms", "ipvm", "ipim"]:
        dataset_samples = list_dir(
            f"{DATASET_ROOT_PATH[dataset_name]}/{dataset_split}",
            suffix=".png",
            contains=filepath_contains,
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = CommonImageDataset(dataset_samples, **args)
    elif dataset_name == "videosham":
        if "attack" not in args:
            print("`attack` not specified. Evaluating on all attacks.")
            dataset_samples = get_all_files(
                f"{DATASET_ROOT_PATH[dataset_name]}",
                suffix=".png",
                contains=filepath_contains,
            )
        else:
            attacks = list(map(int, args["attack"].split()))
            print(f"Evaluating on attacks: {attacks}")
            dataset_samples = []
            for attack in attacks:
                dataset_samples += get_all_files(
                    f"{DATASET_ROOT_PATH[dataset_name]}/attack{attack}",
                    suffix=".png",
                    contains=filepath_contains,
                )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = VideoShamAdobeDataset(dataset_samples, **args)
    elif dataset_name in ["e2fgvi_inpainting", "fuseformer_inpainting"]:
        if "res_divider" not in args:
            print("`res_divider` not specified. Using default value of 1.")
        res_divider = args.get("res_divider", 1)
        resolution = f"ds_{1920 // res_divider}x{1080 // res_divider}"
        dataset_samples = get_all_files(
            f"{DATASET_ROOT_PATH[dataset_name]}/{resolution}/{dataset_split}",
            suffix=(".png", ".jpg"),
            contains=filepath_contains,
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = InpaintingDataset(dataset_samples, **args)
    elif dataset_name == "misl_deepfake":
        print("Warning: MISL Deepfake dataset does not have splits.")
        dataset_samples = list_dir(
            f"{DATASET_ROOT_PATH[dataset_name]}",
            suffix=".png",
            contains=filepath_contains,
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = CommonImageDataset(dataset_samples, **args)
    elif dataset_name == "dfd":
        orig_samples = list_dir(
            f"{DATASET_ROOT_PATH[dataset_name]}/DeepFakeDetection_orig/{dataset_split}", prefix="orig"
        )
        manip_samples = [
            s.replace("mask", "manip")
            for s in list_dir(
                f"{DATASET_ROOT_PATH[dataset_name]}/DeepFakeDetection/{dataset_split}", prefix="mask"
            )
        ]  # filter out all the samples that do not have a mask
        print({"orig": len(orig_samples), "manip": len(manip_samples)})
        dataset_samples = orig_samples + manip_samples
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = DeepfakeDetectionDataset(dataset_samples, **args)
    elif dataset_name in [
        "ffpp_Deepfakes",
        "ffpp_Face2Face",
        "ffpp_FaceSwap",
        "ffpp_NeuralTextures",
    ]:
        ffpp_ds = dataset_name.split("_")[1]
        orig_samples = list_dir(f"{DATASET_ROOT_PATH['ffpp']}/orig/{dataset_split}", prefix="orig")
        manip_samples = [
            s.replace("mask", "manip")
            for s in list_dir(f"{DATASET_ROOT_PATH['ffpp']}/{ffpp_ds}/{dataset_split}", prefix="mask")
        ]  # filter out all the samples that do not have a mask
        print({"orig": len(orig_samples), "manip": len(manip_samples)})
        dataset_samples = orig_samples + manip_samples
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = FaceForensicsPlusPlusDataset(dataset_samples, **args)
    elif dataset_name in ["dfdc", "celeb_df_v2"]:
        dataset_samples = list_dir(
            DATASET_ROOT_PATH[dataset_name],
            suffix=".png",
            contains=filepath_contains,
        )
        if shuffle:
            random.shuffle(dataset_samples)
        if num_samples > 0:
            dataset_samples = dataset_samples[:num_samples]
        dataset = DFDCDataset(dataset_samples, **args)
    else:
        raise NotImplementedError

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
