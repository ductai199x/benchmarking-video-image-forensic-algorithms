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
from torchmetrics.classification import BinaryAUROC as AUROC, BinaryAccuracy as Accuracy
from torchmetrics.functional.classification import binary_f1_score as f1_score, binary_matthews_corrcoef as matthews_corrcoef

from torchvision.transforms import ToTensor
from torchvision.transforms.functional import crop, pad, resize
from tqdm.auto import tqdm

from data.dataset_paths import DATASET_ROOT_PATH
from helper import *
from utils import *
from custom_dataset_classes import *

parser = argparse.ArgumentParser()

available_datasets = [
    "vcms",
    "vpvm",
    "vpim",
    "videosham",
    "e2fgvi_inpainting",
    "fuseformer_inpainting",
    "misl_deepfake",
    "icms",
    "ipvm",
    "ipim",
    "dfd",
    "ffpp_Deepfakes",
    "ffpp_Face2Face",
    "ffpp_FaceSwap",
    "ffpp_NeuralTextures",
    "dfdc",
    "celeb_df_v2"
]

allowed_unknown_args = ["res_divider", "attack"]

exif_ckpt_path = "/media/nas2/trained_models_repository/exifnet_tf1/exif_final.ckpt"
open_proc = lambda cmd_list: subprocess.Popen(
    cmd_list, 
    stdout=subprocess.DEVNULL, 
    stderr=subprocess.DEVNULL, 
    universal_newlines=True, 
    bufsize=1,
)
to_tensor = ToTensor()
is_shuffle = True
random.seed(42)
torch.set_printoptions(precision=15)


def parse_unknown_args(args):
    unknown_args = {}
    key = None
    for arg in args:
        if arg.startswith("--"):
            arg_name = arg[2:]
            if arg_name not in unknown_args:
                unknown_args[arg_name] = ""
                key = arg_name
            else:
                raise ValueError(f"Duplicate argument key {arg_name}")
        else:
            if key is not None:
                unknown_args[key] += arg + " "

    return {k: v.strip() for k, v in unknown_args.items()}


def get_dataset(
    dataset_name,
    dataset_split: Literal["train", "val", "test"] = "test",
    shuffle=False,
    num_samples=-1,
    filepath_contains="",
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
    else:
        raise NotImplementedError

    return dataset_samples


def parse_eval_dataset():
    global parser, subparsers
    p = subparsers.add_parser("dataset", help="Evaluate model on a single dataset.")
    p.add_argument(
        "--dataset_name",
        "--dataset",
        choices=available_datasets,
        type=str,
        required=True,
    )
    p.add_argument(
        "--dataset_split",
        choices=["train", "val", "test"],
        type=str,
        default="test",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=-1,
    )
    p.add_argument(
        "--contains",
        type=str,
        default="",
    )
    p.add_argument(
        "--num_procs",
        type=int,
        default=10,
    )
    p.set_defaults(func=eval_dataset)


def parse_args():
    global parser, subparsers, ARGS
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
        "--results-dir",
        type=str,
        default="exifnet_temp_results",
        help="Results directory. Will be cleaned if --clean-result-dir is enabled.",
    )

    subparsers = parser.add_subparsers()
    parse_eval_dataset()

    ARGS, UNKNOWN_ARGS = parser.parse_known_args()
    UNKNOWN_ARGS = parse_unknown_args(UNKNOWN_ARGS)

    if not set(UNKNOWN_ARGS.keys()).issubset(set(allowed_unknown_args)):
        raise ValueError("Unknown arguments: {}".format(UNKNOWN_ARGS))

    ARGS = argparse.Namespace(**vars(ARGS), **UNKNOWN_ARGS)

    print(ARGS)
    # return

    ARGS.func(ARGS)


def mp_call_eval_single_img(exif_ckpt_path, img_path, temp_results_dir):
    open_proc(
        shlex.split(
            f'python ./models/exifnet/evaluate_single_image.py "{exif_ckpt_path}" "{img_path}" "{temp_results_dir}"'
        )
    ).communicate()


def update_progress_bar(_):
    progress_bar.update()


def eval_dataset(args):
    global temp_results_dir
    temp_results_dir = f"{args.results_dir}/{args.dataset_name}" + (
        f"/attack{args.attack}" if "attack" in args is not None else ""
    )
    if not os.path.exists(temp_results_dir):
        os.makedirs(temp_results_dir)

    dataset = get_dataset(**vars(args))

    if args.compute_metrics is False:
        global progress_bar
        progress_bar = tqdm(total=len(dataset))

        if args.clean_result_dir:
            remove_files_in_dir(temp_results_dir)

        pool = Pool(args.num_procs)
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
        test_loc_f1 = []
        test_loc_mcc = []

        detection_preds = []
        detection_truths = []
        for img_path in tqdm(dataset):
            img_folder, img_basename = os.path.split(os.path.abspath(img_path))
            img_filename, img_extension = os.path.splitext(img_basename)

            if "manip" in img_filename:
                detection_label = 1
            else:
                detection_label = 0

            try:
                meanshift = np.load(f"{temp_results_dir}/{img_filename}.npy")
            except FileNotFoundError:
                continue
            meanshift = np.nan_to_num(meanshift)
            detection_pred = meanshift.mean()
            if detection_label == 1:
                # Load the ground-truth mask
                mask_path = f"{img_folder}/{img_filename.replace('manip', 'mask')}.png" # for dfd and ffpp
                gt_mask = to_tensor(Image.open(mask_path, mode="r"))
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

                pp = pred_mask
                gt = gt_mask

                pp_neg = 1 - pp
                f1_pos = f1_score(pp, gt)
                f1_neg = f1_score(pp_neg, gt)
                if f1_neg > f1_pos:
                    test_loc_f1.append(f1_neg)
                else:
                    test_loc_f1.append(f1_pos)
                
                mcc_pos = matthews_corrcoef(pp, gt)
                mcc_neg = matthews_corrcoef(pp_neg, gt)
                if mcc_neg > mcc_pos:
                    test_loc_mcc.append(mcc_neg)
                else:
                    test_loc_mcc.append(mcc_pos)

            detection_preds.append(detection_pred)
            detection_truths.append(detection_label)

        test_class_acc.update(torch.tensor(detection_preds), torch.tensor(detection_truths))
        test_class_auc.update(torch.tensor(detection_preds), torch.tensor(detection_truths))

        print("test_loc_f1", torch.nan_to_num(torch.tensor(test_loc_f1)).mean())
        print("test_loc_mcc", torch.nan_to_num(torch.tensor(test_loc_mcc)).mean())
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


def main():
    global temp_results_dir
    parse_args()


if __name__ == "__main__":
    main()
