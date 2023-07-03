import sys
import os

from lightning import LightningModule

sys.path.insert(0, os.getcwd())

import argparse
import torch
import yaml

from typing import *

from lightning.pytorch import Trainer

from data.load_single_dataset import load_single_dataset

torch.set_float32_matmul_precision("high")
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

ARCH_CHOICES = [
    "video_transformer",
    "fsg",
    "exif",
    "noiseprint",
    "mvss",
    "mantranet",
]

allowed_unknown_args = ["res_divider", "attack"]


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


def get_model(model_codename: str) -> LightningModule:
    if model_codename == "video_transformer":
        from models.video_transformer import VideoTransformer

        return VideoTransformer
    elif model_codename == "fsg":
        from models.fsg import FSGWholeEvalPLWrapper as FSG

        return FSG
    elif model_codename == "exif":
        print("Exifnet must be evaluated using the `evaluate_exifnet.py` script.")
        exit(1)
    elif model_codename == "noiseprint":
        from models.noiseprint import NoiseprintEvalPLWrapper as Noiseprint

        return Noiseprint
    elif model_codename == "mvss":
        from models.mvssnet import MVSSNetEvalPLWrapper as MVSSNet

        return MVSSNet
    elif model_codename == "mantranet":
        from models.mantranet import ManTraNetEvalPLWrapper as ManTraNet

        return ManTraNet
    else:
        raise NotImplementedError


def get_trainer(args):
    return Trainer(
        accelerator="cpu" if args.cpu else "gpu",
        devices=1,
        logger=False,
        profiler=None,
        callbacks=None,
    )


def eval_dataset(args):
    data = load_single_dataset(**vars(args))
    model = get_model(ARGS.arch)
    trainer = get_trainer(args)
    trainer.test(model, data)


def parse_eval_dataset():
    global parser, subparsers
    p = subparsers.add_parser("dataset", help="Evaluate model on a single dataset.")
    p.add_argument(
        "--dataset_name",
        "--dataset",
        "--name",
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
        "--batch_size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=10,
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
    p.set_defaults(func=eval_dataset)


def parse_args():
    global parser, subparsers, ARGS
    parser.add_argument(
        "--arch",
        type=str,
        choices=ARCH_CHOICES,
        help="The name of the architecture of the model",
        required=True,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu for inference",
    )

    subparsers = parser.add_subparsers()
    parse_eval_dataset()

    ARGS, UNKNOWN_ARGS = parser.parse_known_args()
    UNKNOWN_ARGS = parse_unknown_args(UNKNOWN_ARGS)

    if not set(UNKNOWN_ARGS.keys()).issubset(set(allowed_unknown_args)):
        raise ValueError("Unknown arguments: {}".format(UNKNOWN_ARGS))

    ARGS = argparse.Namespace(**vars(ARGS), **UNKNOWN_ARGS)

    print(ARGS)

    ARGS.func(ARGS)


def main():
    parse_args()


if __name__ == "__main__":
    main()
