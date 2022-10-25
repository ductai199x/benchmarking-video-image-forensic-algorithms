import torch
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import Accuracy, F1Score, AUROC


class MISLNet(torch.nn.Module):
    def __init__(self, num_pre_filters=6):
        super().__init__()
        self.weights_cstr = torch.nn.Parameter(torch.randn(num_pre_filters, 3, 5, 5))

        self.conv1 = torch.nn.Conv2d(
            num_pre_filters, 96, kernel_size=7, stride=2, padding="valid"
        )
        self.bn1 = torch.nn.BatchNorm2d(96, eps=0.001)
        self.tanh1 = torch.nn.Tanh()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv2 = torch.nn.Conv2d(96, 64, kernel_size=5, stride=1, padding="same")
        self.bn2 = torch.nn.BatchNorm2d(64, eps=0.001)
        self.tanh2 = torch.nn.Tanh()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding="same")
        self.bn3 = torch.nn.BatchNorm2d(64, eps=0.001)
        self.tanh3 = torch.nn.Tanh()
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding="same")
        self.bn4 = torch.nn.BatchNorm2d(128, eps=0.001)
        self.tanh4 = torch.nn.Tanh()
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.fc1 = torch.nn.Linear(2 * 2 * 128, 200)
        self.tanh_fc1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(200, 200)
        self.tanh_fc2 = torch.nn.Tanh()

    def initialize_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight.data, 1)
            torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        constr_conv = F.conv2d(x, self.weights_cstr, padding="valid")
        constr_conv = F.pad(constr_conv, (2, 3, 2, 3))

        conv1_out = self.maxpool1(self.tanh1(self.bn1(self.conv1(constr_conv))))
        conv2_out = self.maxpool2(self.tanh2(self.bn2(self.conv2(conv1_out))))
        conv3_out = self.maxpool3(self.tanh3(self.bn3(self.conv3(conv2_out))))
        conv4_out = self.maxpool4(self.tanh4(self.bn4(self.conv4(conv3_out))))

        # tf reshape has differerent order.
        conv4_out = conv4_out.permute(0, 2, 3, 1)
        conv4_out = conv4_out.flatten(1, -1)

        dense1_out = self.tanh_fc1(self.fc1(conv4_out))
        dense2_out = self.tanh_fc2(self.fc2(dense1_out))

        return dense2_out


class MISLnetPLWrapper(LightningModule):
    def __init__(
        self,
        patch_size,
        num_classes,
        lr=1e-3,
        momentum=0.95,
        decay_rate=0.75,
        decay_step=3,
    ):
        super().__init__()
        self.model = MISLNet(num_classes)

        self.lr = lr
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.decay_step = decay_step

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.test_f1 = F1Score()
        self.test_auc = AUROC(num_classes=num_classes, compute_on_step=False)
        self.example_input_array = torch.randn(5, 3, patch_size, patch_size)

        with torch.no_grad():
            self.model(self.example_input_array)
        self.model.apply(self.model.initialize_weights)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.logger is not None:
            self.logger.experiment.add_graph(self, self.example_input_array.cuda())

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)
        self.log(
            "train_acc",
            self.train_acc(logits, y),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss)
        self.log(
            "val_acc",
            self.val_acc(logits, y),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_acc", self.test_acc(logits, y), on_epoch=True, on_step=False)
        self.log("test_f1", self.test_f1(logits, y), on_epoch=True, on_step=False)
        self.log("test_auc", self.test_auc(logits, y), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum
        )
        # steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.decay_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=1, factor=0.5
            ),
            "monitor": "val_acc_epoch",
        }
        return [optimizer], [scheduler]
