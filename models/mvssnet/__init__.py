import lightning.pytorch as pl
import torch
from torchvision.transforms.functional import resize
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef

import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .model.mvss import get_mvss
from .model.resfcn import ResFCN
from .common.tools import inference_single
from .common.utils import calculate_pixel_f1


class MVSSNetWrapper(pl.LightningModule):
    def __init__(self, model_type = 'mvssnet'):
        super().__init__()
        mvssnet_path = '/home/shengbang/benchmarking-video-image-forensic-algorithms/models/mvssnet/ckpt/mvssnet_casia.pt'
        resfcn_path = '/home/shengbang/benchmarking-video-image-forensic-algorithms/models/mvssnet/ckpt/resfcn_casia.pt'
        resize = 512
        th = 0.5
        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = F1Score()
        self.test_loc_mcc = MatthewsCorrCoef(num_classes=2)
        
        if model_type == 'mvssnet':
            self.model = get_mvss(backbone='resnet50',
                            pretrained_base=True,
                            nclass=1,
                            sobel=True,
                            constrain=True,
                            n_input=3,
                            )
            checkpoint = torch.load(mvssnet_path, map_location='cpu')
        elif model_type == 'fcn': 
            self.model = ResFCN()
            checkpoint = torch.load(resfcn_path, map_location='cpu')

        self.model.load_state_dict(checkpoint, strict=True)
        self.model = self.model.to('cuda')
        self.model.eval()
        
    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        # initialize batch predictions:
        detection_preds = []
        localization_preds = []

        for image in x:
            shape = image.shape
            image = resize(image, (512, 512)).permute(1, 2, 0).cpu().numpy()
            # shape should be 512, 512, 3
            image = image[..., ::-1].astype(np.uint8)
            predicted, score = inference_single(img=image, model=self.model, th=0)
            mask_pred = cv2.resize(predicted, (W, H)) / 1.0
            detection_preds.append(float(score))
            localization_preds.append(torch.tensor(mask_pred))
            # print(predicted.shape)
            
        # print(detection_preds, localization_preds)
        self.test_class_acc(
            torch.tensor(detection_preds).to(self.device), y.to(self.device)
        )
        self.test_class_auc(
            torch.tensor(detection_preds).to(self.device), y.to(self.device)
        )
        for i in range(B):
            if y[i] == 0: continue
            loc_pred = localization_preds[i].clone().detach().to(self.device)
            true_mask = m[i].to(self.device)
            self.test_loc_f1(loc_pred, true_mask)
            self.test_loc_mcc(loc_pred, true_mask)    
            
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
        
MVSSNetWrapper = MVSSNetWrapper()