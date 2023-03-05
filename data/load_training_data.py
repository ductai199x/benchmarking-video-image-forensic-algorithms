import sys

sys.path.append("..")

import random
from typing import List

from torch.utils.data import ConcatDataset, DataLoader

from utils import get_all_files, get_filename, list_dir, rand_split

from .common import CommonImageDataset
from .dataset_class_inpaint import InpaintingDataset
from .dataset_class_dfd import DeepfakeDetectionDataset
from .dataset_class_ffpp import FaceForensicsPlusPlusDataset
from .dataset_paths import DATASET_ROOT_PATH


random_seed = 42


def get_dataset(dataset_name: str, dataset_samples: List[str], **args):
    dataset_name = dataset_name.lower()
    if dataset_name in ("vcms", "vpvm", "vpim", "icms", "ipvm", "ipim"):
        return CommonImageDataset(dataset_samples, **args)
    elif dataset_name in ["e2fgvi_inpainting", "fuseformer_inpainting"]:
        return InpaintingDataset(dataset_samples, **args)
    elif dataset_name == "dfd":
        return DeepfakeDetectionDataset(dataset_samples, **args)
    elif dataset_name == "ffpp":
        return FaceForensicsPlusPlusDataset(dataset_samples, **args)
    else:
        raise NotImplementedError
    # elif dataset_name == "videosham":
    #     return VideoShamAdobeDataset(dataset_samples, **args)
    # elif dataset_name == "e2fgvi_inpainting":
    #     return E2fgviDavisDataset(dataset_samples, **args)
    # else:
    #     raise NotImplementedError


def load_training_data(dataset_name, batch_size=3, num_workers=10):
    function_name = f"load_data_{dataset_name}"
    if function_name in globals():
        dataset_function = globals()[function_name]
        train_ds, val_ds = dataset_function()
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return train_dl, val_dl
    else:
        raise ValueError(f"{dataset_name} does not exists. Please recheck.")


def load_data_finetune_inpainting():
    train_vids_n_imgs_ds, _ = load_data_finetune_all_videos_and_images()
    train_e2fgvi_ds, val_e2fgvi_ds = load_data_e2fgvi_inpainting()
    train_fuseformer_ds, val_fuseformer_ds = load_data_fuseformer_inpainting()

    print(
        {
            "train_vids_n_imgs_ds": len(train_vids_n_imgs_ds),
            "train_e2fgvi_ds": len(train_e2fgvi_ds),
            "train_fuseformer_ds": len(train_fuseformer_ds),
            "val_e2fgvi_ds": len(val_e2fgvi_ds),
            "val_fuseformer_ds": len(val_fuseformer_ds),
        }
    )

    train_ds = ConcatDataset([train_vids_n_imgs_ds, train_e2fgvi_ds, train_fuseformer_ds])
    val_ds = ConcatDataset([val_e2fgvi_ds, val_fuseformer_ds])

    return train_ds, val_ds


def load_data_finetune_dfd():
    train_vids_n_imgs_ds, _ = load_data_finetune_all_videos_and_images()
    train_dfd_ds, val_dfd_ds = load_data_dfd()

    print(
        {
            "train_vids_n_imgs_ds": len(train_vids_n_imgs_ds),
            "train_dfd_ds": len(train_dfd_ds),
            "val_dfd_ds": len(val_dfd_ds),
        }
    )

    train_ds = ConcatDataset([train_vids_n_imgs_ds, train_dfd_ds])
    val_ds = ConcatDataset([val_dfd_ds])

    return train_ds, val_ds


def load_data_finetune_ffpp():
    train_vids_n_imgs_ds, _ = load_data_finetune_all_videos_and_images()
    train_ffpp_ds, val_ffpp_ds = load_data_ffpp()

    print(
        {
            "train_vids_n_imgs_ds": len(train_vids_n_imgs_ds),
            "train_ffpp_ds": len(train_ffpp_ds),
            "val_ffpp_ds": len(val_ffpp_ds),
        }
    )

    train_ds = ConcatDataset([train_vids_n_imgs_ds, train_ffpp_ds])
    val_ds = ConcatDataset([val_ffpp_ds])

    return train_ds, val_ds


def load_data_finetune_ffpp_dfd():
    train_vids_n_imgs_ds, _ = load_data_finetune_all_videos_and_images()
    train_ffpp_ds, val_ffpp_ds = load_data_ffpp()
    train_dfd_ds, val_dfd_ds = load_data_dfd()

    # train_ffpp_ds, _ = rand_split(train_ffpp_ds, 0.75, random_seed)
    # train_dfd_ds, _ = rand_split(train_dfd_ds, 0.75, random_seed)
    # val_ffpp_ds, _ = rand_split(val_ffpp_ds, 0.5, random_seed)
    # val_dfd_ds, _ = rand_split(val_dfd_ds, 0.5, random_seed)

    print(
        {
            "train_vids_n_imgs_ds": len(train_vids_n_imgs_ds),
            "train_ffpp_ds": len(train_ffpp_ds),
            "train_dfd_ds": len(train_dfd_ds),
            "val_ffpp_ds": len(val_ffpp_ds),
            "val_dfd_ds": len(val_dfd_ds),
        }
    )

    train_ds = ConcatDataset([train_vids_n_imgs_ds, train_ffpp_ds, train_dfd_ds])
    val_ds = ConcatDataset([val_ffpp_ds, val_dfd_ds])

    return train_ds, val_ds


def load_data_finetune_all_videos_and_images():
    train_vcms_ds, val_vcms_ds = load_data_vcms()
    train_vpvm_ds, val_vpvm_ds = load_data_vpvm()
    train_vpim_ds, val_vpim_ds = load_data_vpim()
    train_icms_ds, val_icms_ds = load_data_icms()
    train_ipvm_ds, val_ipvm_ds = load_data_ipvm()
    train_ipim_ds, val_ipim_ds = load_data_ipim()

    retain_ratio = 0.03

    train_vcms_ds, _ = rand_split(train_vcms_ds, retain_ratio, random_seed)
    train_vpvm_ds, _ = rand_split(train_vpvm_ds, retain_ratio, random_seed)
    train_vpim_ds, _ = rand_split(train_vpim_ds, retain_ratio, random_seed)
    val_vcms_ds, _ = rand_split(val_vcms_ds, retain_ratio, random_seed)
    val_vpvm_ds, _ = rand_split(val_vpvm_ds, retain_ratio, random_seed)
    val_vpim_ds, _ = rand_split(val_vpim_ds, retain_ratio, random_seed)

    train_icms_ds, _ = rand_split(train_icms_ds, retain_ratio, random_seed)
    train_ipvm_ds, _ = rand_split(train_ipvm_ds, retain_ratio, random_seed)
    train_ipim_ds, _ = rand_split(train_ipim_ds, retain_ratio, random_seed)
    val_icms_ds, _ = rand_split(val_icms_ds, retain_ratio, random_seed)
    val_ipvm_ds, _ = rand_split(val_ipvm_ds, retain_ratio, random_seed)
    val_ipim_ds, _ = rand_split(val_ipim_ds, retain_ratio, random_seed)

    train_ds = ConcatDataset(
        [
            train_vcms_ds,
            train_vpvm_ds,
            train_vpim_ds,
            train_icms_ds,
            train_ipvm_ds,
            train_ipim_ds,
        ]
    )
    val_ds = ConcatDataset([val_vcms_ds, val_vpvm_ds, val_vpim_ds, val_icms_ds, val_ipvm_ds, val_ipim_ds])

    return train_ds, val_ds


def load_data_mixed_all_videos():
    train_vcms_ds, val_vcms_ds = load_data_vcms()
    train_vpvm_ds, val_vpvm_ds = load_data_vpvm()
    train_vpim_ds, val_vpim_ds = load_data_vpim()

    train_vcms_ds, _ = rand_split(train_vcms_ds, 0.1, random_seed)
    train_vpvm_ds, _ = rand_split(train_vpvm_ds, 0.1, random_seed)
    train_vpim_ds, _ = rand_split(train_vpim_ds, 0.8, random_seed)
    val_vcms_ds, _ = rand_split(val_vcms_ds, 0.1, random_seed)
    val_vpvm_ds, _ = rand_split(val_vpvm_ds, 0.1, random_seed)
    val_vpim_ds, _ = rand_split(val_vpim_ds, 0.8, random_seed)

    train_ds = ConcatDataset([train_vcms_ds, train_vpvm_ds, train_vpim_ds])
    val_ds = ConcatDataset([val_vcms_ds, val_vpvm_ds, val_vpim_ds])

    return train_ds, val_ds


def load_data_mixed_all_videos_and_images():
    train_vcms_ds, val_vcms_ds = load_data_vcms()
    train_vpvm_ds, val_vpvm_ds = load_data_vpvm()
    train_vpim_ds, val_vpim_ds = load_data_vpim()
    train_icms_ds, val_icms_ds = load_data_icms()
    train_ipvm_ds, val_ipvm_ds = load_data_ipvm()
    train_ipim_ds, val_ipim_ds = load_data_ipim()

    train_vcms_ds, _ = rand_split(train_vcms_ds, 0.1, random_seed)
    train_vpvm_ds, _ = rand_split(train_vpvm_ds, 0.1, random_seed)
    train_vpim_ds, _ = rand_split(train_vpim_ds, 0.3, random_seed)
    val_vcms_ds, _ = rand_split(val_vcms_ds, 0.1, random_seed)
    val_vpvm_ds, _ = rand_split(val_vpvm_ds, 0.1, random_seed)
    val_vpim_ds, _ = rand_split(val_vpim_ds, 0.3, random_seed)

    train_icms_ds, _ = rand_split(train_icms_ds, 0.1, random_seed)
    train_ipvm_ds, _ = rand_split(train_ipvm_ds, 0.1, random_seed)
    train_ipim_ds, _ = rand_split(train_ipim_ds, 0.7, random_seed)
    val_icms_ds, _ = rand_split(val_icms_ds, 0.1, random_seed)
    val_ipvm_ds, _ = rand_split(val_ipvm_ds, 0.1, random_seed)
    val_ipim_ds, _ = rand_split(val_ipim_ds, 0.7, random_seed)

    train_ds = ConcatDataset(
        [
            train_vcms_ds,
            train_vpvm_ds,
            train_vpim_ds,
            train_icms_ds,
            train_ipvm_ds,
            train_ipim_ds,
        ]
    )
    val_ds = ConcatDataset([val_vcms_ds, val_vpvm_ds, val_vpim_ds, val_icms_ds, val_ipvm_ds, val_ipim_ds])

    return train_ds, val_ds


###################################################################


def load_data_e2fgvi_inpainting():
    train_inpainting_samples = get_all_files(
        f"{DATASET_ROOT_PATH['e2fgvi_inpainting']}/ds_1920x1080/train", suffix=".png"
    )
    val_inpainting_samples = get_all_files(
        f"{DATASET_ROOT_PATH['e2fgvi_inpainting']}/ds_1920x1080/val", suffix=".png"
    )

    train_ds = get_dataset("e2fgvi_inpainting", train_inpainting_samples)
    val_ds = get_dataset("e2fgvi_inpainting", val_inpainting_samples)

    return train_ds, val_ds


def load_data_fuseformer_inpainting():
    train_inpainting_samples = get_all_files(
        f"{DATASET_ROOT_PATH['fuseformer_inpainting']}/ds_1920x1080/train", suffix=".png"
    )
    val_inpainting_samples = get_all_files(
        f"{DATASET_ROOT_PATH['fuseformer_inpainting']}/ds_1920x1080/val", suffix=".png"
    )

    train_ds = get_dataset("fuseformer_inpainting", train_inpainting_samples)
    val_ds = get_dataset("fuseformer_inpainting", val_inpainting_samples)

    return train_ds, val_ds


def load_data_ffpp():
    ffpp_subsets = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    subset_split_ratio = 1 / len(ffpp_subsets)
    train_ffpp_manip_samples = []
    val_ffpp_manip_samples = []
    for subset in ffpp_subsets:
        train_manip_samples = [
            s.replace("mask", "manip")
            for s in list_dir(f"{DATASET_ROOT_PATH['ffpp']}/{subset}/train", prefix="mask")
        ]
        train_manip_samples, _ = rand_split(train_manip_samples, subset_split_ratio, random_seed)
        val_manip_samples = [
            s.replace("mask", "manip")
            for s in list_dir(f"{DATASET_ROOT_PATH['ffpp']}/{subset}/val", prefix="mask")
        ]
        val_manip_samples, _ = rand_split(val_manip_samples, subset_split_ratio, random_seed)

        train_ffpp_manip_samples += train_manip_samples
        val_ffpp_manip_samples += val_manip_samples

    train_ffpp_orig_samples = list_dir(f"{DATASET_ROOT_PATH['ffpp']}/orig/train", prefix="orig")
    val_ffpp_orig_samples = list_dir(f"{DATASET_ROOT_PATH['ffpp']}/orig/val", prefix="orig")

    train_ffpp_samples = train_ffpp_orig_samples + train_ffpp_manip_samples
    val_ffpp_samples = val_ffpp_orig_samples + val_ffpp_manip_samples

    train_ds = get_dataset("ffpp", train_ffpp_samples)
    val_ds = get_dataset("ffpp", val_ffpp_samples)

    # ratio = 100 * 1000 / len(train_ds)
    ratio = 0.1
    train_ds, _ = rand_split(train_ds, ratio, random_seed)
    val_ds, _ = rand_split(val_ds, ratio, random_seed)

    return train_ds, val_ds


def load_data_dfd():
    random.seed(42)
    train_dfd_orig_samples = list_dir(
        f"{DATASET_ROOT_PATH['dfd']}/DeepFakeDetection_orig/train", prefix="orig"
    )
    train_dfd_manip_samples = [
        s.replace("mask", "manip")
        for s in list_dir(f"{DATASET_ROOT_PATH['dfd']}/DeepFakeDetection/train", prefix="mask")
    ]  # filter out all the samples that do not have a mask
    train_dfd_manip_samples = random.sample(train_dfd_manip_samples, k=len(train_dfd_orig_samples))
    train_dfd_samples = train_dfd_orig_samples + train_dfd_manip_samples

    val_dfd_orig_samples = list_dir(f"{DATASET_ROOT_PATH['dfd']}/DeepFakeDetection_orig/val", prefix="orig")
    val_dfd_manip_samples = [
        s.replace("mask", "manip")
        for s in list_dir(f"{DATASET_ROOT_PATH['dfd']}/DeepFakeDetection/val", prefix="mask")
    ]  # filter out all the samples that do not have a mask
    val_dfd_manip_samples = random.sample(val_dfd_manip_samples, k=len(val_dfd_orig_samples))
    val_dfd_samples = val_dfd_orig_samples + val_dfd_manip_samples

    train_ds = get_dataset("dfd", train_dfd_samples)
    val_ds = get_dataset("dfd", val_dfd_samples)

    # ratio = 100 * 1000 / len(train_ds)
    ratio = 0.1
    train_ds, _ = rand_split(train_ds, ratio, random_seed)
    val_ds, _ = rand_split(val_ds, ratio, random_seed)

    return train_ds, val_ds


def load_data_vcms():
    train_vcms_samples = list_dir(f"{DATASET_ROOT_PATH['vcms']}/train", suffix=".png")
    val_vcms_samples = list_dir(f"{DATASET_ROOT_PATH['vcms']}/val", suffix=".png")

    train_ds = get_dataset("vcms", train_vcms_samples)
    val_ds = get_dataset("vcms", val_vcms_samples)

    return train_ds, val_ds


def load_data_vpvm():
    train_vpvm_samples = list_dir(f"{DATASET_ROOT_PATH['vpvm']}/train", suffix=".png")
    val_vpvm_samples = list_dir(f"{DATASET_ROOT_PATH['vpvm']}/val", suffix=".png")

    train_ds = get_dataset("vpvm", train_vpvm_samples)
    val_ds = get_dataset("vpvm", val_vpvm_samples)

    return train_ds, val_ds


def load_data_vpim():
    train_vpim_samples = list_dir(f"{DATASET_ROOT_PATH['vpim']}/train", suffix=".png")
    val_vpim_samples = list_dir(f"{DATASET_ROOT_PATH['vpim']}/val", suffix=".png")

    train_ds = get_dataset("vpim", train_vpim_samples)
    val_ds = get_dataset("vpim", val_vpim_samples)

    return train_ds, val_ds


def load_data_icms():
    train_icms_samples = list_dir(f"{DATASET_ROOT_PATH['icms']}/train", suffix=".png")
    val_icms_samples = list_dir(f"{DATASET_ROOT_PATH['icms']}/val", suffix=".png")

    train_ds = get_dataset("icms", train_icms_samples)
    val_ds = get_dataset("icms", val_icms_samples)

    return train_ds, val_ds


def load_data_ipvm():
    train_ipvm_samples = list_dir(f"{DATASET_ROOT_PATH['ipvm']}/train", suffix=".png")
    val_ipvm_samples = list_dir(f"{DATASET_ROOT_PATH['ipvm']}/val", suffix=".png")

    train_ds = get_dataset("ipvm", train_ipvm_samples)
    val_ds = get_dataset("ipvm", val_ipvm_samples)

    return train_ds, val_ds


def load_data_ipim():
    train_ipim_samples = list_dir(f"{DATASET_ROOT_PATH['ipim']}/train", suffix=".png")
    val_ipim_samples = list_dir(f"{DATASET_ROOT_PATH['ipim']}/val", suffix=".png")

    train_ds = get_dataset("ipim", train_ipim_samples)
    val_ds = get_dataset("ipim", val_ipim_samples)

    return train_ds, val_ds
