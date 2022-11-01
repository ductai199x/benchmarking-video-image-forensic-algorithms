import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
from PIL import Image

import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

from models.exifnet.demo import Demo as Exifnet

exif_ckpt_path = sys.argv[1]
img_path = sys.argv[2]
temp_results_dir = sys.argv[3]

img_folder, img_basename = os.path.split(os.path.abspath(img_path))
img_filename, img_extension = os.path.splitext(img_basename)

if "manip" in img_path:
    label = 1
else:
    label = 0


img = Image.open(img_path, mode="r")
img = np.array(img).astype(np.float32)

exifnet = Exifnet(
    exif_ckpt_path,
    use_gpu=0,
    quality=1,
    num_per_dim=10,
)
meanshift = exifnet.run(
    img,
    use_ncuts=False,
    blue_high=True,
)

save_result_path = f"{temp_results_dir}/{img_filename}.npy"

np.save(save_result_path, meanshift)