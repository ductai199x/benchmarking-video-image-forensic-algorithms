import lightning.pytorch as pl
import torch
from torchvision.transforms.functional import resize, normalize
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef

import numpy as np

from .model.mvss import get_mvss
# from .common.tools import inference_single


mvssnet_path = '/media/nas2/trained_models_repository/mvssnet_pytorch/mvssnet_casia.pt'
resfcn_path = '/media/nas2/trained_models_repository/mvssnet_pytorch/resfcn_casia.pt'

normalize_dict = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class MVSSNetImageEvalWrapper(pl.LightningModule):
    def __init__(
        self,
        mvssnet_path,
        resfcn_path,
    ):
        super().__init__()
        
        self.resize = (512, 512)
        self.th = 0.5
        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = F1Score()
        self.test_loc_mcc = MatthewsCorrCoef(num_classes=2)
        
        self.model = get_mvss(
            backbone='resnet50',
            pretrained_base=True,
            nclass=1,
            sobel=True,
            constrain=True,
            n_input=3,
        )
        checkpoint = torch.load(mvssnet_path, map_location='cpu')

        self.model.load_state_dict(checkpoint, strict=True)
        self.model = self.model.to(self.device).eval()
        
    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        x = resize(x, self.resize)
        normalize(x, normalize_dict["mean"], normalize_dict["std"], inplace=True)

        _, pred_mask = self.model(x)
        pred_mask = torch.sigmoid(pred_mask).detach()
        if torch.isnan(pred_mask).any() or torch.isinf(pred_mask).any():
            pred_labels = torch.zeros(B)
        else:
            pred_labels = pred_mask.flatten(1, -1).max(dim=1)[0]
        pred_mask = resize(pred_mask, (H, W)).squeeze().float()

        self.test_class_acc.update(
            pred_labels.to(self.device), y.to(self.device)
        )
        self.test_class_auc.update(
            pred_labels.to(self.device), y.to(self.device)
        )
        self.test_loc_f1.update(pred_mask[y==1], m[y==1])
        self.test_loc_mcc.update(pred_mask[y==1], m[y==1])
                
            
    def on_test_epoch_end(self) -> None:
        self.log("test_loc_f1", self.test_loc_f1.compute())
        self.log("test_loc_mcc", self.test_loc_mcc.compute())
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


MVSSNetImageEvalWrapper = MVSSNetImageEvalWrapper(
    mvssnet_path=mvssnet_path,
    resfcn_path=resfcn_path,
)