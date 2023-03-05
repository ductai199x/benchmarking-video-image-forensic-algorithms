import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def process_single_video(model, dataloader: DataLoader, output_dir: str, use_cpu: bool = False):
    device = "cpu" if use_cpu else "cuda"
    model = model.to(device)

    class_preds = []
    with torch.no_grad():
        for frame_batch, frame_idxs in tqdm(dataloader):
            class_out, patch_out = model(frame_batch.to(device))
            class_out, patch_out = class_out.detach().cpu(), patch_out.detach().cpu()

            class_out = torch.softmax(class_out, dim=1)

            patch_preds = [
                model.patch_to_pixel_pred(
                    pl,
                    model.patch_size,
                    model.img_size,
                    min_thresh=0.1,
                    max_num_regions=3,
                    final_thresh=0.21,
                )
                for pl in patch_out
            ]
            pixel_preds = torch.vstack([pp.get_pixel_preds().unsqueeze(0) for pp in patch_preds])

            for score, frame_idx, frame, pixel_preds in zip(class_out, frame_idxs, frame_batch, pixel_preds):
                fig, ax = plt.subplots(figsize=(8, 10))
                ax.imshow(frame.permute(1, 2, 0).to(torch.uint8))
                ax.imshow(pixel_preds, cmap="gray", alpha=0.25, vmin=0.0, vmax=1.0)
                ax.set_title(f"Frame Index = {frame_idx} - Predicted Label = {score[1]:5f}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.savefig(f"{output_dir}/{frame_idx:05d}.png", dpi=150, bbox_inches="tight")
                plt.close()

            class_preds.append(class_out[:, 1])
    class_preds = torch.concat(class_preds)
    return {
        "mean": float(class_preds.mean()),
        "median": float(class_preds.median()),
        "std": float(class_preds.std()),
        "90_conf_intv": [
            float(class_preds.mean() - class_preds.std() * 1.645),
            float(class_preds.mean() + class_preds.std() * 1.645),
        ],
    }
