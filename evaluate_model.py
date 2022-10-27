import argparse
import random
from typing import *

import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from custom_dataset_classes import *
from helper import get_all_files

torchmetrics_version = list(map(int, torchmetrics.__version__.split(".")))
if torchmetrics_version > [0, 7, 0]:
    raise InterruptedError(
        "torchmetrics version > 0.7.0 may provide inaccurate MCC results."
    )

parser = argparse.ArgumentParser()

NUM_WORKERS = 8
SHUFFLE = True
torch.manual_seed(42)

DS_CHOICES = [
    "image_adv_splc",
    "image_vis_aug",
    "image_invis_aug",
    "video_adv_splc",
    "video_vis_aug",
    "video_invis_aug",
    "video_sham_adobe",
    "video_e2fgvi_davis",
]

ARCH_CHOICES = [
    "video_transformer",
    "fsg",
    "exif",
    "noiseprint",
]


def get_model(model_codename: str) -> LightningModule:
    if model_codename == "video_transformer":
        from models.video_transformer import VideoTransformer

        return VideoTransformer
    elif model_codename == "fsg":
        from models.fsg import FSGWholeImageEvalPLWrapper as FSG

        return FSG
    elif model_codename == "exif":
        from models.exifnet import ExifnetImageEvalPLWrapper as Exifnet

        return Exifnet
    elif model_codename == "noiseprint":
        from models.noiseprint import NoiseprintImageEvalPLWrapper as Noiseprint

        return Noiseprint
    else:
        raise NotImplementedError


def get_trainer():
    return Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=-1,
        enable_model_summary=False,
        logger=None,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        fast_dev_run=False,
    )


def get_dataset(ds_choice: str) -> DataLoader:
    if ds_choice == "image_adv_splc":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/image_cam_model_splicing/test",
            suffix=".png",
        ),
        test_dl = DataLoader(
            GenericImageDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "image_vis_aug":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/image_visible_aug/test", suffix=".png"
        )
        test_dl = DataLoader(
            GenericImageDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "image_invis_aug":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/image_invisible_aug_super_low_SSIM_loss/val",
            suffix=".png",
        )
        test_dl = DataLoader(
            GenericImageDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "video_adv_splc":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/video_advanced_splicing/test", suffix=".png"
        )
        test_dl = DataLoader(
            GenericImageDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "video_vis_aug":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/video_visible_aug/test", suffix=".png"
        )
        test_dl = DataLoader(
            GenericImageDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "video_invis_aug":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/video_invisible_aug/test", suffix=".png"
        )
        test_dl = DataLoader(
            GenericImageDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "video_sham_adobe":
        img_files = get_all_files(
            "/media/nas2/Datasets/VideoSham-adobe-research/extracted_frames_ge_1920x1080"
            + (f"/attack{ARGS.adobesham_attack}" if ARGS.adobesham_attack > 0 else ""),
            suffix=".png",
        )
        test_dl = DataLoader(
            VideoShamAdobeDataset(img_files),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    elif ds_choice == "video_e2fgvi_davis":
        resolution_divider = 1
        resolution = (1080 // resolution_divider, 1920 // resolution_divider)
        img_files = get_all_files(
            f"/media/nas2/Tai/13-e2fgvi-video-inpainting/ds_{resolution[1]}x{resolution[0]}",
            suffix=(".png", ".jpg"),
        )
        test_dl = DataLoader(
            E2fgviDavisDataset(img_files, resolution=(1080, 1920)),
            batch_size=ARGS.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
        )
    else:
        raise (NotImplementedError)

    return test_dl


def run_test(trainer: Trainer, model: LightningModule, test_dl: DataLoader) -> dict:
    return trainer.test(model, test_dl)


def parse_args():
    global ARGS
    parser.add_argument(
        "--arch",
        type=str,
        choices=ARCH_CHOICES,
        help="The name of the architecture of the model",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DS_CHOICES,
        help="The name of the dataset that you want to benchmark on",
        required=True,
    )
    parser.add_argument(
        "--adobesham-attack",
        type=int,
        choices=[-1, 1, 2, 3, 4],
        help="The specific attack from adobesham that you want to benchmark on. -1 is ALL",
        default=-1,
    )
    parser.add_argument(
        "--e2fgvi-res-div",
        type=int,
        choices=[1, 2, 3, 4],
        help="The specific resolution divider for e2fgvi that you want to benchmark on",
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size for the dataloader",
        default=10,
    )
    ARGS = parser.parse_args()


def main():
    parse_args()

    model = get_model(ARGS.arch)
    trainer = get_trainer()
    dataset = get_dataset(ARGS.dataset)
    result_dict = run_test(trainer, model, dataset)
    print(result_dict)


if __name__ == "__main__":
    main()
