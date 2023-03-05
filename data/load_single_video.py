import random
import decord
from torch.utils.data import DataLoader


def load_single_video(
    video_path,
    shuffle,
    max_num_samples,
    sample_every,
    batch_size,
    num_workers,
):
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    batch_idxs = list(range(0, len(vr), sample_every))
    if shuffle:
        random.shuffle(batch_idxs)
    if max_num_samples > 0:
        batch_idxs = batch_idxs[:max_num_samples]

    batch_idxs = sorted(batch_idxs)
    frame_batch = vr.get_batch(batch_idxs)
    frame_batch = frame_batch.permute(0, 3, 1, 2).float()

    return DataLoader(
        list(zip(frame_batch, batch_idxs)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
