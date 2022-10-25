import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, F1Score

from .mislnet import MISLNet
from .comparenet import CompareNet


class FSG(torch.nn.Module):
    def __init__(
        self,
        num_pre_filters=6,
        input_dim=200,
        map1_dim=2048,
        map2_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.mislnet = MISLNet(num_pre_filters)
        self.comparenet = CompareNet(input_dim, map1_dim, map2_dim)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        x1 = self.mislnet(x1)
        x2 = self.mislnet(x2)

        out = self.comparenet([x1, x2])

        return out


class FSGPLWrapper(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()

        self.model = FSG(**config)

        self.train_acc, self.val_acc, self.test_acc = Accuracy(), Accuracy(), Accuracy()
        self.test_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_f1 = F1Score()

        self.patch_size = config["patch_size"]
        self.lr = config["lr"] or 1e-5
        self.decay_step = config["decay_step"] or 2
        self.decay_rate = config["decay_rate"] or 0.85
        self.save_hyperparameters(config)
        self.example_input_array = torch.randn(
            2, 3, 3, self.patch_size[0], self.patch_size[1]
        )  # xBCHW

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.logger.experiment.add_graph(self, self.example_input_array.cuda())

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log(
            "train_acc", self.train_acc, on_epoch=True, on_step=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("test_loss", loss)
        self.test_acc(logits, y)
        self.test_auc(logits, y)
        self.test_f1(logits, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

        self.log("test_auc", self.test_auc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.95)
        steplr = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.decay_rate
        )
        return [optimizer], [steplr]
