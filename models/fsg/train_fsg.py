import os
import sys

sys.path.insert(0, os.getcwd())


import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, IterableDataset

from helper import get_all_files
from fsg import FSGPLWrapper

patch_size = 128
train_tfrecs_path = f"/media/nas2/misl_image_db_70_class/train/{patch_size}"
val_tfrecs_path = f"/media/nas2/misl_image_db_70_class/val/{patch_size}"

training_records = get_all_files(train_tfrecs_path, suffix="tfrecord")
validating_records = get_all_files(val_tfrecs_path, suffix="tfrecord")

n_classes = 70
batch_size = n_classes * 2

AUTOTUNE = tf.data.experimental.AUTOTUNE

sample_description = {
    "label": tf.io.FixedLenFeature([], tf.int64),
    "raw": tf.io.FixedLenFeature([], tf.string),
}


def _parse_sample(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    img_features = tf.io.parse_single_example(example_proto, sample_description)
    image = tf.io.parse_tensor(img_features["raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [patch_size, patch_size, 3])
    label = tf.cast(img_features["label"], tf.uint8)
    return image, label


def cartesian_product(a, b):
    # all combinations of elements in a and b, with a first and b second
    # From jdehesa in https://stackoverflow.com/questions/53123763/how-to-perform-cartesian-product-with-tensorflow
    c = tf.stack(tf.meshgrid(a, b, indexing="ij"), axis=-1)
    c = tf.reshape(c, (-1, 2))
    return c


def combinations(a, b):
    # all unique combinations of elements in a and b
    # order matters (e.g. [1 2] is considered unique from [2 1], so we get both)
    # removes any instances where a and b are the same. we shouldn't ever encounter a scenario
    # where we need to know whether the same patch is forensically same/different
    # tested with single values (indices), not sure if it would work on more complex tensors, though it should

    c1 = cartesian_product(a, b)
    c2 = cartesian_product(b, a)
    c = tf.concat([c1, c2], 0)

    inds_diff = tf.where(
        tf.not_equal(c[:, 1], c[:, 0])
    )  # first and second column are different
    inds_diff = tf.squeeze(inds_diff)  # remove size 1 dimensions
    out = tf.gather(c, inds_diff)  # return only combinations that are different

    return out


def random_pairings(X, y, n_same=64, n_diff=64, shuffle=True):
    # indices of the patches
    inds = tf.range(tf.shape(y)[0])

    # all pairings of indices, and corresponding class labels y (with no pairings of the same patch)
    icombs = combinations(inds, inds)
    ycombs = tf.gather(y, icombs)

    # indices of same and different camera model
    inds_diff = tf.where(
        tf.not_equal(ycombs[:, 1], ycombs[:, 0])
    )  # first and second column are different
    inds_diff = tf.squeeze(inds_diff)  # remove size 1 dimensions
    # n_diff_available = tf.shape(inds_diff)[0]  # number of available "different" pairs

    inds_same = tf.where(
        tf.equal(ycombs[:, 1], ycombs[:, 0])
    )  # first and second column are the same
    inds_same = tf.squeeze(inds_same)  # remove size 1 dimensions
    # n_same_available = tf.shape(inds_same)[0]  # number of available "same" pairs

    # randomly select n combinations. How do we make sure there are enough
    # combinations/n isn't too big?.
    # I guess right now we need to be careful to choose a batch size that is big enough.
    rinds_same = tf.random.shuffle(inds_same)[:n_same]
    rinds_diff = tf.random.shuffle(inds_diff)[:n_diff]

    rinds = tf.concat(
        [rinds_same, rinds_diff], 0
    )  # combine indices of randomly chosen same and different combinations
    if shuffle:
        rinds = tf.random.shuffle(rinds)  # shuffle up the rinds
    ricombs = tf.gather(icombs, rinds)  # convert to pairs of X/y indices

    X_pair = tf.gather(X, ricombs)  # gather patch pairs
    labels_pair = tf.gather(y, ricombs)  # gather label pairs
    y_out = tf.equal(
        labels_pair[:, 1], labels_pair[:, 0]
    )  # tell me when the pairs have the same label

    y_out = tf.cast(y_out, tf.int32)  # same label = 1, different label = 0
    # y_out = tf.stack([y_out, 1-y_out], axis=1)
    return X_pair, y_out


raw_train_set = tf.data.Dataset.from_tensor_slices(training_records).interleave(
    lambda x: tf.data.TFRecordDataset(x).map(
        _parse_sample, num_parallel_calls=AUTOTUNE
    ),
    num_parallel_calls=AUTOTUNE,
    cycle_length=n_classes,
    block_length=1,
)

raw_val_set = tf.data.Dataset.from_tensor_slices(validating_records).interleave(
    lambda x: tf.data.TFRecordDataset(x).map(
        _parse_sample, num_parallel_calls=AUTOTUNE
    ),
    num_parallel_calls=AUTOTUNE,
    cycle_length=n_classes,
    block_length=1,
)

train_tfds = (
    raw_train_set.batch(batch_size=batch_size)
    .map(lambda X, y: random_pairings(X, y, 64, 64, True))
    .prefetch(32)
)
val_tfds = (
    raw_val_set.batch(batch_size=batch_size)
    .map(lambda X, y: random_pairings(X, y, 64, 64, False))
    .prefetch(32)
)


class MyIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def process_data(self, generator):
        for image_pair, label in generator:
            image_pair = torch.from_numpy(image_pair.numpy()).permute(
                1, 0, 4, 2, 3
            )  # BxHWC->xBCHW
            label = torch.from_numpy(label.numpy()).long()
            yield image_pair, label

    def get_stream(self, generator):
        return self.process_data(generator)

    def __iter__(self):
        return self.get_stream(self.generator)


train_itds = MyIterableDataset(train_tfds)
val_itds = MyIterableDataset(val_tfds)


train_dl = DataLoader(train_itds, batch_size=None, num_workers=0)
val_dl = DataLoader(val_itds, batch_size=None, num_workers=0)


config = {
    "patch_size": (128, 128),
    "lr": 2e-5,
    "decay_step": 2,
    "decay_rate": 0.75,
    "num_pre_filters": 6,
    "input_dim": 200,
    "map1_dim": 2048,
    "map2_dim": 64,
}

model_name = "fsg_image"

fsg_wrapper = FSGPLWrapper(**config)
fsg_wrapper.model.load_state_dict(torch.load("fsg-image-pytorch-from-tf1.pt"))
fsg_wrapper.model.eval()
# torch.manual_seed(0)
# batch_size = 2
# x = torch.randint(0, 255, (batch_size, 2, 3, 128, 128)).float()
# x1 = torch.randint(0, 255, (batch_size, 3, 128, 128)).float()
# x2 = torch.randint(0, 255, (batch_size, 3, 128, 128)).float()
# print(fsg_wrapper(x))
# exit(0)

prev_ckpt = None
# prev_ckpt = "/home/tai/1-workdir/6-image-fsg-pytorch/lightning_logs/fsg_image/version_1/checkpoints/fsg_image=0-epoch=53-val_acc_epoch=0.8682.ckpt"
resume = False

if prev_ckpt:
    fsg_wrapper = FSGPLWrapper.load_from_checkpoint(prev_ckpt, **config)
else:
    fsg_wrapper = FSGPLWrapper(**config)


version = 3
monitor_metric = "val_acc_epoch"
tb_log_path = f"src/lightning_logs/{model_name}"
ckpt_path = f"{tb_log_path}/version_{version}/checkpoints"

logger = TensorBoardLogger(
    save_dir=os.getcwd(), version=version, name=tb_log_path, log_graph=True
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
model_ckpt = ModelCheckpoint(
    dirpath=ckpt_path,
    monitor=monitor_metric,
    filename=f"{{{model_name}}}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
    verbose=True,
    save_last=True,
    mode="max",
)

trainer = Trainer(
    gpus=1,
    max_epochs=-1,
    resume_from_checkpoint=prev_ckpt if resume else None,
    enable_model_summary=True,
    logger=logger,
    # profiler="pytorch",
    callbacks=[TQDMProgressBar(refresh_rate=1), lr_monitor, model_ckpt],
    fast_dev_run=False,
)

fsg_wrapper(
    fsg_wrapper.example_input_array
)  # to get the model's graph to display in tblogger

# trainer.fit(fsg_wrapper, train_dl, val_dl)
trainer.test(fsg_wrapper, val_dl)
