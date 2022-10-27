import pytorch_lightning as pl
import tensorflow as tf
import torch
from torchvision.transforms.functional import rgb_to_grayscale, resize
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

from .network import FullConvNet
from .noiseprint import genNoiseprint
from .noiseprint_blind import noiseprint_blind_post


QF = 101
model_name = "net"
chkpt_folder = "/media/nas2/trained_models_repository/noiseprint_tf1/%s_jpg%d/model"


class NoiseprintImageEvalPLWrapper(pl.LightningModule):
    def __init__(
        self,
        chkpt_folder,
        model_name="net",
        QF=101,
    ):
        super().__init__()

        self.QF = QF if QF < 100 else 101

        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = F1Score()
        self.test_loc_mcc = MatthewsCorrCoef(num_classes=2)

        self.model_name = model_name
        self.chkpt_folder = chkpt_folder
        self.chkpt_fname = self.chkpt_folder % (self.model_name, self.QF)

        tf.compat.v1.reset_default_graph()
        self.x_data = tf.compat.v1.placeholder(
            tf.float32, [1, None, None, 1], name="x_data"
        )
        self.net = FullConvNet(self.x_data, 0.9, tf.constant(False), num_levels=17)
        self.saver = tf.compat.v1.train.Saver(self.net.variables_list)

        self.sess = tf.compat.v1.Session()
        self.saver.restore(self.sess, self.chkpt_fname)

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        # initialize batch predictions:
        detection_preds = []
        localization_preds = []

        for image in x:
            image = (
                rgb_to_grayscale(image).float().permute(1, 2, 0).squeeze(-1) / 255.0
            ).cpu()
            residual = genNoiseprint(
                self.sess,
                self.net,
                self.x_data,
                image,
                self.QF,
                model_name="net",
            )
            loc_pixel_map, _, _, _, _, _ = noiseprint_blind_post(residual, image)
            if loc_pixel_map is None:
                detection_preds.append(0.0)
                localization_preds.append(torch.zeros(H, W))
            else:
                loc_pixel_map = torch.tensor(loc_pixel_map).detach().nan_to_num(0.0)
                amplitude = loc_pixel_map.max() - loc_pixel_map.min()
                if amplitude > 1e-10:
                    normalized_loc_map = (
                        loc_pixel_map - loc_pixel_map.min()
                    ) / amplitude
                else:
                    normalized_loc_map = loc_pixel_map - loc_pixel_map.min()
                normalized_loc_map = resize(
                    normalized_loc_map.unsqueeze(0), [H, W]
                ).squeeze()

                detection_preds.append(loc_pixel_map.mean())
                localization_preds.append(normalized_loc_map)

        self.test_class_acc(
            torch.tensor(detection_preds).to(self.device), y.to(self.device)
        )
        self.test_class_auc(
            torch.tensor(detection_preds).to(self.device), y.to(self.device)
        )
        for i in range(B):
            loc_pred = localization_preds[i].clone().detach().to(self.device)
            true_mask = m[i].to(self.device)
            self.test_loc_f1(loc_pred, true_mask)
            self.test_loc_mcc(loc_pred, true_mask)

    def on_test_epoch_end(self) -> None:
        self.sess.close()
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
        self.log("test_class_fnr", neg_preds.sum() / neg_labels.sum())


NoiseprintImageEvalPLWrapper = NoiseprintImageEvalPLWrapper(
    chkpt_folder,
    model_name,
    QF,
)
