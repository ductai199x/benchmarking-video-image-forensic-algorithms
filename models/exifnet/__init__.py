import lightning.pytorch as pl
import tensorflow as tf
import torch
from torchmetrics.classification import BinaryAUROC as AUROC, BinaryAccuracy as Accuracy
from torchmetrics.functional.classification import binary_f1_score as f1_score, binary_matthews_corrcoef as matthews_corrcoef

from .demo import Demo as Exifnet

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.disable_eager_execution()

ckpt_path = "/media/nas2/trained_models_repository/exifnet_tf1/exif_final.ckpt"


class ExifnetEvalPLWrapperClass(pl.LightningModule):
    def __init__(
        self,
        ckpt_path,
        use_gpu=0,
        quality=1.0,
        num_per_dim=10,
        **kwargs,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.use_gpu = use_gpu
        self.quality = quality
        self.num_per_dim = num_per_dim
        self.model = Exifnet(self.ckpt_path, self.use_gpu, self.quality, self.num_per_dim)

        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = []
        self.test_loc_mcc = []

    def forward(self, x):
        B, C, H, W = x.shape

        # initialize batch predictions:
        detection_preds = []
        localization_masks = []

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
            else:
                loc_pixel_map = meanshift
            loc_pixel_map = torch.tensor(loc_pixel_map > 0.25).to(torch.uint8)
            loc_pixel_map = loc_pixel_map / 1.0

            detection_preds.append(detection_pred)
            localization_masks.append(loc_pixel_map)
        
        localization_masks = torch.concat([torch.tensor(l).unsqueeze(0) for l in localization_masks], dim=0)
        return torch.tensor(detection_preds), torch.tensor(localization_masks)

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        detection_preds, localization_masks = self(x)

        self.test_class_acc(
            detection_preds.to(self.device), y.to(self.device)
        )
        self.test_class_auc(
            detection_preds.to(self.device), y.to(self.device)
        )
        for i in range(B):
            if y[i] == 0: continue
            pp = localization_masks[i].to(self.device)
            gt = m[i].to(self.device)

            pp_neg = 1 - pp
            f1_pos = f1_score(pp, gt)
            f1_neg = f1_score(pp_neg, gt)
            if f1_neg > f1_pos:
                self.test_loc_f1.append(f1_neg)
            else:
                self.test_loc_f1.append(f1_pos)
            
            mcc_pos = matthews_corrcoef(pp, gt, num_classes=2)
            mcc_neg = matthews_corrcoef(pp_neg, gt, num_classes=2)
            if mcc_neg > mcc_pos:
                self.test_loc_mcc.append(mcc_neg)
            else:
                self.test_loc_mcc.append(mcc_pos)

    def on_test_epoch_end(self) -> None:
        self.log("test_loc_f1", torch.nan_to_num(torch.tensor(self.test_loc_f1)).mean())
        self.log("test_loc_mcc", torch.nan_to_num(torch.tensor(self.test_loc_mcc)).mean())
        self.log("test_class_auc", self.test_class_auc.compute())
        self.log("test_class_acc", self.test_class_acc.compute())

        self.test_class_probs = torch.concat(
            [preds_batch for preds_batch in self.test_class_auc.preds]
        )
        self.test_class_preds = torch.concat(
            [(preds_batch > 0.5).int() for preds_batch in self.test_class_auc.preds]
        )
        self.test_class_truths = torch.concat(
            [truths_batch for truths_batch in self.test_class_auc.target]
        )

        pos_labels = self.test_class_truths == 1
        pos_preds = self.test_class_preds[pos_labels] == 1
        neg_labels = self.test_class_truths == 0
        neg_preds = self.test_class_preds[neg_labels] == 0
        self.log("test_class_tpr", pos_preds.sum() / pos_labels.sum())
        self.log("test_class_tnr", neg_preds.sum() / neg_labels.sum())


ExifnetEvalPLWrapper = ExifnetEvalPLWrapperClass(
    ckpt_path,
    use_gpu=0,
    quality=1.0,
    num_per_dim=10,
)
