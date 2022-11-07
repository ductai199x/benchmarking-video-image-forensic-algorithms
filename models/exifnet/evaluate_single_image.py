import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import crop, pad, resize

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

from models.exifnet.demo import Demo as Exifnet

to_tensor = ToTensor()

exif_ckpt_path = sys.argv[1]
img_path = sys.argv[2]
temp_results_dir = sys.argv[3]

img_folder, img_basename = os.path.split(os.path.abspath(img_path))
img_filename, img_extension = os.path.splitext(img_basename)

if "manip" in img_path:
    label = 1
else:
    label = 0


img = to_tensor(Image.open(img_path, mode="r")) * 255
img = img.float()[0:3]
if img.shape[1] != 1080:
    img = crop(img, 0, 0, 1080, 1920)
img = img.permute(1,2,0).numpy()

exifnet = Exifnet(
    exif_ckpt_path,
    use_gpu=0,
    quality=1,
    num_per_dim=10,
)
try:
    meanshift, _ = exifnet.run_vote(
        img
    )
except:
    meanshift = exifnet.run(
        img,
        use_ncuts=False,
        blue_high=True,
    )

save_result_path = f"{temp_results_dir}/{img_filename}.npy"

np.save(save_result_path, meanshift)
