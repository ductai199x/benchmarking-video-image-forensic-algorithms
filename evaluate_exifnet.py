# Due to the fact that their EXIFnet code has a memory leak -> run out of both CPU RAM and GPU VRAM after some time, I made this file so that we can just run exifnet per-image, get the statistics, then exit, or destroy the process completely so as to prevent OOM errors

import argparse
import random
import shlex
import subprocess
from multiprocessing import Pool
from typing import *

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import crop, pad, resize
from tqdm.auto import tqdm

from custom_dataset_classes import *
from helper import *

parser = argparse.ArgumentParser()

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

exif_ckpt_path = "/media/nas2/trained_models_repository/exifnet_tf1/exif_final.ckpt"
open_proc = lambda cmd_list: subprocess.Popen(
    cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1
)
to_tensor = ToTensor()
is_shuffle = True
random.seed(42)
torch.set_printoptions(precision=15)


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
            + (f"/attack{ARGS.adobesham_attack}" if ARGS.adobesham_attack > 0 else ""),
            suffix=".png",
        )
    elif ds_choice == "video_e2fgvi_davis":
        resolution_divider = ARGS.e2fgvi_res_div
        resolution = (1080 // resolution_divider, 1920 // resolution_divider)
        img_files = get_all_files(
            f"/media/nas2/Tai/13-e2fgvi-video-inpainting/ds_{resolution[1]}x{resolution[0]}",
            suffix=(".png", ".jpg"),
        )
    else:
        raise (NotImplementedError)

    return img_files


def parse_args():
    global ARGS
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Enable this flag to start computing metrics using the results from the temporary results folder",
    )
    parser.add_argument(
        "--clean-result-dir",
        action="store_true",
        help="Enable this flag to clean temp result directory. Won't do anything with --compute-metrics enabled.",
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
        "--num-procs",
        type=int,
        help="The number of multiprocessing processes to evaluate multiple images in the dataset at the same time",
        default=4,
    )
    ARGS = parser.parse_args()


def mp_call_eval_single_img(exif_ckpt_path, img_path, temp_results_dir):
    open_proc(
        shlex.split(
            f'python ./models/exifnet/evaluate_single_image.py "{exif_ckpt_path}" "{img_path}" "{temp_results_dir}"'
        )
    ).communicate()


def update_progress_bar(_):
    progress_bar.update()


def main():
    global temp_results_dir
    parse_args()

    temp_results_dir = f"{os.getcwd()}/exifnet_temp_results/{ARGS.dataset}" + (
        f"/attack{ARGS.adobesham_attack}" if ARGS.adobesham_attack > 0 else ""
    )
    if not os.path.exists(temp_results_dir):
        os.makedirs(temp_results_dir)
    dataset = get_dataset(ARGS.dataset)
    if is_shuffle:
        random.shuffle(dataset)

    if ARGS.compute_metrics is False:
        global progress_bar
        progress_bar = tqdm(total=len(dataset))

        if ARGS.clean_result_dir:
            remove_files_in_dir(temp_results_dir)

        pool = Pool(ARGS.num_procs)
        for img_path in dataset:
            img_folder, img_basename = os.path.split(os.path.abspath(img_path))
            img_filename, img_extension = os.path.splitext(img_basename)
            if os.path.exists(f"{temp_results_dir}/{img_filename}.npy"): 
                update_progress_bar(None)
                continue
            pool.apply_async(
                mp_call_eval_single_img,
                args=(exif_ckpt_path, img_path, temp_results_dir),
                callback=update_progress_bar,
            )
        pool.close()
        pool.join()

    else:
        # initialize torchmetrics objects
        test_class_acc = Accuracy()
        test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        test_loc_f1 = F1Score()
        test_loc_mcc = MatthewsCorrCoef(num_classes=2)

        detection_preds = []
        detection_truths = []
        for img_path in tqdm(dataset):
            img_folder, img_basename = os.path.split(os.path.abspath(img_path))
            img_filename, img_extension = os.path.splitext(img_basename)

            if "manip" in img_filename:
                detection_label = 1
            else:
                detection_label = 0

            meanshift = np.load(f"{temp_results_dir}/{img_filename}.npy")
            meanshift = np.nan_to_num(meanshift)
            detection_pred = meanshift.mean()
            if detection_label == 1:
                # Load the ground-truth mask
                gt_mask = to_tensor(Image.open(f"{img_folder}/{img_filename}.mask", mode="r"))
                if len(gt_mask.shape) > 2:
                    gt_mask = gt_mask[0]
                if gt_mask.shape[0] != 1080:
                    gt_mask = crop(gt_mask, 0, 0, 1080, 1920)
                
                gt_mask[gt_mask > 0] = 1
                gt_mask = gt_mask.int()

                # Produce localization map from the meanshift heatmap
                amplitude = meanshift.max() - meanshift.min()
                if amplitude > 1e-10:
                    pred_mask = (meanshift - meanshift.min()) / amplitude
                else:
                    pred_mask = meanshift
                pred_mask = torch.tensor(pred_mask > 0.25).to(torch.float)

                test_loc_f1.update(pred_mask, gt_mask)
                test_loc_mcc.update(pred_mask, gt_mask)

            detection_preds.append(detection_pred)
            detection_truths.append(detection_label)

        test_class_acc.update(torch.tensor(detection_preds), torch.tensor(detection_truths))
        test_class_auc.update(torch.tensor(detection_preds), torch.tensor(detection_truths))

        print("test_loc_f1", test_loc_f1.compute())
        print("test_loc_mcc", test_loc_mcc.compute())
        print("test_class_auc", test_class_auc.compute())
        print("test_class_acc", test_class_acc.compute())

        test_class_probs = torch.concat([preds_batch for preds_batch in test_class_auc.preds])
        test_class_preds = torch.concat([(preds_batch > 0.5).int() for preds_batch in test_class_auc.preds])
        test_class_truths = torch.concat([truths_batch for truths_batch in test_class_auc.target])

        pos_labels = test_class_truths == 1
        pos_preds = test_class_preds[pos_labels] == 1
        neg_labels = test_class_truths == 0
        neg_preds = test_class_preds[neg_labels] == 0
        print("test_class_tpr", pos_preds.sum() / pos_labels.sum())
        print("test_class_tnr", neg_preds.sum() / neg_labels.sum())


if __name__ == "__main__":
    main()
