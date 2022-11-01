# Due to the fact that their EXIFnet code has a memory leak -> run out of both CPU RAM and GPU VRAM after some time, I made this file so that we can just run exifnet per-image, get the statistics, then exit, or destroy the process completely so as to prevent OOM errors

import argparse
import random
import shlex
import subprocess
from typing import *

import numpy as np
import torch
import torchmetrics
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import crop, pad, resize
from tqdm.auto import tqdm

from custom_dataset_classes import *
from helper import *

DS_CHOICE = "video_adv_splc"
ADOBESHAM_ATTACK = -1
temp_results_dir = f"{os.getcwd()}/exifnet_temp_results/{DS_CHOICE}" + (
    f"/attack{ADOBESHAM_ATTACK}" if ADOBESHAM_ATTACK > 0 else ""
)
exif_ckpt_path = "/media/nas2/trained_models_repository/exifnet_tf1/exif_final.ckpt"

open_proc = lambda cmd_list: subprocess.Popen(
    cmd_list, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1
)


def get_dataset(ds_choice: str) -> DataLoader:
    if ds_choice == "image_adv_splc":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/image_cam_model_splicing/test",
            suffix=".png",
        )
    elif ds_choice == "image_vis_aug":
        img_files = get_all_files("/media/nas2/graph_sim_data/image_visible_aug/test", suffix=".png")
    elif ds_choice == "image_invis_aug":
        img_files = get_all_files(
            "/media/nas2/graph_sim_data/image_invisible_aug_super_low_SSIM_loss/val",
            suffix=".png",
        )
    elif ds_choice == "video_adv_splc":
        img_files = get_all_files("/media/nas2/graph_sim_data/video_advanced_splicing/test", suffix=".png")
    elif ds_choice == "video_vis_aug":
        img_files = get_all_files("/media/nas2/graph_sim_data/video_visible_aug/test", suffix=".png")
    elif ds_choice == "video_invis_aug":
        img_files = get_all_files("/media/nas2/graph_sim_data/video_invisible_aug/test", suffix=".png")
    elif ds_choice == "video_sham_adobe":
        img_files = get_all_files(
            "/media/nas2/Datasets/VideoSham-adobe-research/extracted_frames_ge_1920x1080"
            + (f"/attack{ADOBESHAM_ATTACK}" if ADOBESHAM_ATTACK > 0 else ""),
            suffix=".png",
        )
    elif ds_choice == "video_e2fgvi_davis":
        resolution_divider = 1
        resolution = (1080 // resolution_divider, 1920 // resolution_divider)
        img_files = get_all_files(
            f"/media/nas2/Tai/13-e2fgvi-video-inpainting/ds_{resolution[1]}x{resolution[0]}",
            suffix=(".png", ".jpg"),
        )
    else:
        raise (NotImplementedError)

    return img_files


def main():

    if not os.path.exists(temp_results_dir):
        os.makedirs(temp_results_dir)

    remove_files_in_dir(temp_results_dir)

    dataset = get_dataset(DS_CHOICE)
    for img_path in tqdm(dataset):
        open_proc(
            shlex.split(
                f'python ./models/exifnet/evaluate_single_image.py "{exif_ckpt_path}" "{img_path}" "{temp_results_dir}" > /dev/null'
            )
        ).communicate()


if __name__ == "__main__":
    main()
