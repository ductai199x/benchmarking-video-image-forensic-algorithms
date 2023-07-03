import lightning.pytorch as pl
import torch
from torchmetrics.classification import BinaryAUROC as AUROC, BinaryAccuracy as Accuracy
from torchmetrics.functional.classification import (
    binary_f1_score as f1_score,
    binary_matthews_corrcoef as matthews_corrcoef,
)

from .modelCore import load_pretrain_model_by_index


pretrained_dir = "/media/nas2/trained_models_repository/mantranet_tf1"
model_index = 4


class ManTraNetEvalPLWrapperClass(pl.LightningModule):
    def __init__(
        self,
        pretrained_dir,
        model_index,
        **kwargs,
    ):
        super().__init__()
        self.model = load_pretrain_model_by_index(model_index, pretrained_dir)

        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = []
        self.test_loc_mcc = []

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).cpu().numpy()
        localization_masks = torch.tensor(self.model.predict(x)).squeeze()
        detection_preds = localization_masks.mean(dim=(1, 2))

        return detection_preds, localization_masks

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        detection_preds, localization_masks = self(x)

        self.test_class_acc.update(detection_preds.to(self.device), y.to(self.device))
        self.test_class_auc.update(detection_preds.to(self.device), y.to(self.device))
        for i in range(B):
            if y[i] == 0:
                continue
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

    def on_test_epoch_end(self, outputs):
        self.log("test_loc_f1", torch.nan_to_num(torch.tensor(self.test_loc_f1)).mean())
        self.log("test_loc_mcc", torch.nan_to_num(torch.tensor(self.test_loc_mcc)).mean())
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
        self.log("test_class_tnr", neg_preds.sum() / neg_labels.sum())


ManTraNetEvalPLWrapper = ManTraNetEvalPLWrapperClass(pretrained_dir, model_index)
