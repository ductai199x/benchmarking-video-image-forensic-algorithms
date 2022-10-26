import pytorch_lightning as pl
import tensorflow as tf
import torch
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef

from .demo import Demo as Exifnet

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.disable_eager_execution()

ckpt_path = "/media/nas2/trained_models_repository/exifnet_tf1/exif_final.ckpt"


class ExifnetImageEvalPLWrapper(pl.LightningModule):
    def __init__(
        self,
        ckpt_path,
        use_gpu=0,
        quality=1.0,
        num_per_dim=10,
        **kwargs,
    ):
        super().__init__()
        self.model = Exifnet(ckpt_path, use_gpu, quality, num_per_dim)

        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = F1Score()
        self.test_loc_mcc = MatthewsCorrCoef(num_classes=2)

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        # initialize batch predictions:
        detection_preds = []
        localization_preds = []

        for image in x:
            meanshift = self.model.run(
                image.permute(1, 2, 0).cpu().numpy(), 
                use_ncuts=False, 
                blue_high=True,
            )

            detection_pred = meanshift.mean()
            amplitude = meanshift.max() - meanshift.min()
            if amplitude > 1e-10:
                loc_pixel_map = (meanshift - meanshift.min()) / amplitude
            loc_pixel_map = torch.tensor(loc_pixel_map > 0.25).to(torch.uint8)
            loc_pixel_map = loc_pixel_map / 1.0

            detection_preds.append(detection_pred)
            localization_preds.append(loc_pixel_map)

        self.test_class_acc(torch.tensor(detection_preds).to(self.device), y.to(self.device))
        self.test_class_auc(torch.tensor(detection_preds).to(self.device), y.to(self.device))
        for i in range(B):
            loc_pred = localization_preds[i].clone().detach().to(self.device)
            true_mask = m[i].to(self.device)
            self.test_loc_f1(loc_pred, true_mask)
            self.test_loc_mcc(loc_pred, true_mask)

    def on_test_epoch_end(self) -> None:
        self.log("test_loc_f1", self.test_loc_f1.compute())
        self.log("test_loc_mcc", self.test_loc_mcc.compute())
        self.log("test_class_auc", self.test_class_auc.compute())
        self.log("test_class_acc", self.test_class_acc.compute())

        self.test_class_probs = torch.concat([preds_batch for preds_batch in self.test_class_auc.preds])
        self.test_class_preds = torch.concat(
            [(preds_batch > 0.5).int() for preds_batch in self.test_class_auc.preds]
        )
        self.test_class_truths = torch.concat([truths_batch for truths_batch in self.test_class_auc.target])

        pos_labels = self.test_class_truths == 1
        pos_preds = self.test_class_preds[pos_labels] == 1
        neg_labels = self.test_class_truths == 0
        neg_preds = self.test_class_preds[neg_labels] == 0
        self.log("test_class_tpr", pos_preds.sum() / pos_labels.sum())
        self.log("test_class_fnr", neg_preds.sum() / neg_labels.sum())


ExifnetImageEvalPLWrapper = ExifnetImageEvalPLWrapper(
    ckpt_path,
    use_gpu=0,
    quality=1.0,
    num_per_dim=10,
)
