from typing import *

import torch
import torchvision
from pytorch_lightning.core.module import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef

from .long_dist_attn import LongDistanceAttention
from .mislnet import MISLnetPLWrapper
from .patch_predictions import PatchPredictions
from .xception import Xception, XceptionPLWrapper


class SpatialAttentionPretrainedFE(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_forg_template=3,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        bb1_db_depth=1,
        fe="mislnet",
        fe_config={},
        fe_ckpt="",
        fe_freeze=True,
        **kwargs
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.fe_name = fe.lower()
        self.fe_freeze = fe_freeze

        if self.fe_name == "mislnet":
            self.pretrained_fe = MISLnetPLWrapper()
        elif self.fe_name == "xception":
            self.pretrained_fe = XceptionPLWrapper()
        else:
            raise (NotImplementedError)

        if fe_ckpt is not None and len(fe_ckpt) > 0:
            self.pretrained_fe = self.pretrained_fe.load_from_checkpoint(
                fe_ckpt, **fe_config
            )

        if self.fe_name == "mislnet":
            self.pretrained_fe = nn.Sequential(
                *(list(self.pretrained_fe.model.children())[0][:-2])
            )
            self.pretrained_fe_con = None  # connector layer
        elif self.fe_name == "xception":
            self.pretrained_fe = nn.Sequential(
                *(list(self.pretrained_fe.model.children())[:-1])
            )
            self.pretrained_fe_con = nn.Linear(2048, 200)

        self.backbone = Xception(in_chans, embed_dim, bb1_db_depth)

        self.bb_condense = nn.Linear(embed_dim, embed_dim - 200)

        self.long_dist_attn = LongDistanceAttention(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            num_forg_template,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            act_layer,
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(2),
        )

        self.localizer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 32, 1, kernel_size=(1, 1), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # split image into non-overlapping patches
        kernel_size, stride = self.patch_size, self.patch_size
        patches = (
            x.unfold(2, kernel_size, stride)
            .unfold(3, kernel_size, stride)
            .permute(0, 2, 3, 1, 4, 5)
        )
        # gather up all the patch into a large batch
        patches = patches.contiguous().view(-1, 3, kernel_size, kernel_size)
        # feed large batch into the pretrained feature extractor
        if self.fe_freeze:
            with torch.no_grad():
                fe = self.pretrained_fe(patches)
        else:
            fe = self.pretrained_fe(patches)

        if self.pretrained_fe_con:
            if self.fe_name == "xception":
                fe = F.adaptive_avg_pool2d(fe, (1, 1))
                fe = fe.view(fe.size(0), -1)
                fe = self.pretrained_fe_con(fe)
        # feed large batch into the backbone to produce their features
        bb = self.backbone(patches)
        bb = self.bb_condense(bb)
        # concatinate fe and bb into a single vector of features
        bb_fe = torch.cat([bb, fe], dim=1)
        # split large batch back into embedded images
        bb_fe = bb_fe.contiguous().view(B, -1, self.embed_dim)
        # get the maps from the long distance attention
        lda_maps = self.long_dist_attn(bb_fe)
        # scale the embeddings by the attention maps
        scaled_bb_fe = torch.einsum("ijk,ilj->iljk", bb_fe, lda_maps)
        # scaled_bb = scaled_bb.contiguous().view(B, -1)
        scaled_bb_fe = torch.einsum("iljk->ijk", scaled_bb_fe)

        # feed the scaled embeddings to a classifier to get the output
        class_label = self.classifier(scaled_bb_fe)
        patch_label = self.localizer(
            scaled_bb_fe.permute(0, 2, 1).unsqueeze(-1)
        )  # for localization
        patch_label = patch_label.view(B, -1)
        return class_label, patch_label


class SpatialAttnPretrainedFEPLWrapper(LightningModule):
    def __init__(self, **config):
        super().__init__()

        self.model = SpatialAttentionPretrainedFE(**config)
        self.train_class_acc = Accuracy()
        self.val_class_acc = Accuracy()
        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = F1Score()
        self.test_loc_mcc = MatthewsCorrCoef(num_classes=2)

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.loss_alpha = config["loss_alpha"]
        self.lr = config["lr"] or 1e-5
        self.decay_step = config["decay_step"] or 2
        self.decay_rate = config["decay_rate"] or 0.85
        self.save_hyperparameters(config)
        self.example_input_array = torch.randn(2, 3, 1080, 1920)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.logger is not None:
            self.logger.experiment.add_graph(
                self, self.example_input_array.to(self.device)
            )

    def get_patch_pred(self, m):
        batch_size = m.shape[0]
        kernel_size, stride = self.patch_size, self.patch_size
        p = m.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        p = p.contiguous().view(-1, kernel_size, kernel_size)
        p = torch.flatten(p, start_dim=1, end_dim=2)
        p = torch.sum(p, dim=1) / (kernel_size * kernel_size)
        p = p.view(batch_size, -1)
        return p

    def get_pixel_pred(self, p):
        batch_size = p.shape[0]
        kernel_size, stride = self.patch_size, self.patch_size
        # TODO: change this to reflect stride < kernel_size
        m_p_size0, m_p_size1 = (
            kernel_size * (self.img_size[0] // kernel_size),
            kernel_size * (self.img_size[1] // kernel_size),
        )
        m_p = p.unsqueeze(-1).repeat(1, 1, kernel_size * kernel_size).permute(0, 2, 1)
        m_p = F.fold(
            m_p,
            output_size=(m_p_size0, m_p_size1),
            kernel_size=kernel_size,
            stride=stride,
        ).squeeze()
        return m_p, m_p_size0, m_p_size1

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        p = self.get_patch_pred(m)

        class_logits, patch_logits = self(x.float())
        class_loss = F.cross_entropy(class_logits, y)
        patch_loss = F.binary_cross_entropy(patch_logits, p)

        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * patch_loss

        with torch.no_grad():
            if self.logger is not None and self.global_step % 1000 == 0:
                m_p, m_p_size0, m_p_size1 = self.get_pixel_pred(patch_logits)
                m = m[:, 0:m_p_size0, 0:m_p_size1]
                sample_imgs = torch.cat([m_p.unsqueeze(1), m.unsqueeze(1)], dim=-2)
                grid = torchvision.utils.make_grid(
                    sample_imgs, padding=10, pad_value=255
                )
                self.logger.experiment.add_image(
                    "generated_masks", grid, self.global_step
                )

            self.log("train_loss", loss)
            self.log("train_class_loss", class_loss)
            self.log("train_loc_loss", patch_loss)
            self.train_class_acc(class_logits, y)
            self.log(
                "train_class_acc",
                self.train_class_acc,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m = batch
        p = self.get_patch_pred(m)

        class_logits, patch_logits = self(x.float())
        class_loss = F.cross_entropy(class_logits, y)
        patch_loss = F.binary_cross_entropy(patch_logits, p)

        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * patch_loss

        with torch.no_grad():
            self.log("val_loss", loss)
            self.log("val_class_loss", class_loss)
            self.log("val_loc_loss", patch_loss)
            self.val_class_acc(class_logits, y)
            self.log(
                "val_class_acc",
                self.val_class_acc,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        p = self.get_patch_pred(m)

        class_logits, patch_logits = self(x.float())
        class_loss = F.cross_entropy(class_logits, y)
        patch_loss = F.binary_cross_entropy(patch_logits, p)

        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * patch_loss

        with torch.no_grad():
            self.log("test_loss", loss, on_epoch=True, on_step=False)
            self.log("test_class_loss", class_loss, on_epoch=True, on_step=False)
            self.log("test_loc_loss", patch_loss, on_epoch=True, on_step=False)
            self.test_class_acc(class_logits, y)

            if y.sum() > 0:
                # TODO: implement some gaussian smoothing function instead of this
                manip_img_patch_logits = patch_logits[y == 1].cpu()
                patch_preds = [
                    PatchPredictions(
                        pl, self.patch_size, self.img_size, max_num_regions=2
                    )
                    for pl in manip_img_patch_logits
                ]
                pixel_preds = torch.vstack(
                    [pp.get_pixel_preds().unsqueeze(0) for pp in patch_preds]
                )

                m_h, m_w = pixel_preds.shape[1], pixel_preds.shape[2]
                true_mask = m[y == 1, :m_h, :m_w].to(torch.uint8)

                self.test_loc_f1(pixel_preds.to(self.device), true_mask.to(self.device))
                self.test_loc_mcc(
                    pixel_preds.to(self.device), true_mask.to(self.device)
                )
            # only compute auc at the end
            self.test_class_auc(class_logits, y)

    def on_test_epoch_end(self) -> None:
        self.log("test_loc_f1", self.test_loc_f1.compute())
        self.log("test_loc_mcc", self.test_loc_mcc.compute())
        self.log("test_class_auc", self.test_class_auc.compute())
        self.log("test_class_acc", self.test_class_acc.compute())

        self.test_class_probs = torch.concat(
            [torch.softmax(preds, dim=1)[:, 1] for preds in self.test_class_auc.preds]
        )
        self.test_class_preds = torch.concat(
            [
                torch.argmax(torch.softmax(preds, dim=1), dim=1)
                for preds in self.test_class_auc.preds
            ]
        )
        self.test_class_truths = torch.concat(
            [truths for truths in self.test_class_auc.target]
        )

        pos_labels = self.test_class_truths == 1
        pos_preds = self.test_class_preds[pos_labels] == 1
        neg_labels = self.test_class_truths == 0
        neg_preds = self.test_class_preds[neg_labels] == 0
        self.log("test_class_tpr", pos_preds.sum() / pos_labels.sum())
        self.log("test_class_fnr", neg_preds.sum() / neg_labels.sum())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.97)
        steplr = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.decay_rate
        )
        return [optimizer], [steplr]


### Old function for the old dataset. keeping it here just in case
# def get_patch_pred(self, y):
#     c, x0, x1, y0, y1 = torch.split(y.T, 1, dim=0)
#     c, x0, x1, y0, y1 = [i.squeeze(0) for i in [c, x0, x1, y0, y1]]
#     batch_size = y.shape[0]
#     masks = torch.zeros(batch_size, self.img_size[0], self.img_size[1])
#     for i in range(len(masks)):
#         if c[i] == 1:
#             masks[i][x0[i]: x1[i], y0[i]: y1[i]] = 1

#     p = self.get_patch_pred(masks)
#     p = p.view(batch_size, -1).to(x.device)
#     kernel_size, stride = self.patch_size, self.patch_size
#     p = masks.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
#     p = p.contiguous().view(-1, kernel_size, kernel_size)
#     p = torch.flatten(p, start_dim=1, end_dim=2)
#     p = torch.sum(p, dim=1) / (kernel_size*kernel_size)
#     return p
