import lightning.pytorch as pl
import tensorflow as tf
import torch
from torchvision.transforms.functional import rgb_to_grayscale, resize
from torchmetrics.classification import BinaryAUROC as AUROC, BinaryAccuracy as Accuracy
from torchmetrics.functional.classification import binary_f1_score as f1_score, binary_matthews_corrcoef as matthews_corrcoef

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

from .network import FullConvNet
from .noiseprint import genNoiseprint
from .noiseprint_blind import noiseprint_blind_post
from .post_em import getSpamFromNoiseprint


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
        self.test_loc_f1 = []
        self.test_loc_mcc = []

        self.model_name = model_name
        self.chkpt_folder = chkpt_folder
        self.chkpt_fname = self.chkpt_folder % (self.model_name, self.QF)

        tf.compat.v1.reset_default_graph()
        self.x_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 1], name="x_data")
        self.net = FullConvNet(self.x_data, 0.9, tf.constant(False), num_levels=17)
        self.saver = tf.compat.v1.train.Saver(self.net.variables_list)

        self.sess = tf.compat.v1.Session()
        self.saver.restore(self.sess, self.chkpt_fname)

    def get_features(self, image):
        image = (rgb_to_grayscale(image).float().permute(1, 2, 0).squeeze(-1) / 255.0).cpu()
        residual = genNoiseprint(
            self.sess,
            self.net,
            self.x_data,
            image,
            self.QF,
        )
        spam = getSpamFromNoiseprint(residual, image)
        return spam

    def forward(self, x):
        B, C, H, W = x.shape
        # initialize batch predictions:
        detection_preds = []
        localization_masks = []

        for image in x:
            image = (rgb_to_grayscale(image).float().permute(1, 2, 0).squeeze(-1) / 255.0).cpu()
            residual = genNoiseprint(
                self.sess,
                self.net,
                self.x_data,
                image,
                self.QF,
            )
            loc_pixel_map, _, _, _, _, _ = noiseprint_blind_post(residual, image)
            if loc_pixel_map is None:
                detection_preds.append(0.0)
                localization_masks.append(torch.zeros(H, W))
            else:
                loc_pixel_map = torch.tensor(loc_pixel_map).detach().nan_to_num(0.0)
                amplitude = loc_pixel_map.max() - loc_pixel_map.min()
                if amplitude > 1e-10:
                    normalized_loc_map = (loc_pixel_map - loc_pixel_map.min()) / amplitude
                else:
                    normalized_loc_map = loc_pixel_map - loc_pixel_map.min()
                normalized_loc_map = resize(normalized_loc_map.unsqueeze(0), [H, W]).squeeze()

                # normalized_loc_map[normalized_loc_map >= 0.10] = 1.0
                # normalized_loc_map[normalized_loc_map < 0.10] = 0.0

                detection_preds.append(loc_pixel_map.mean())
                localization_masks.append(normalized_loc_map)

        localization_masks = torch.concat([torch.tensor(l).unsqueeze(0) for l in localization_masks], dim=0)
        return torch.tensor(detection_preds), torch.tensor(localization_masks)

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        detection_preds, localization_masks = self(x)

        self.test_class_acc(detection_preds.to(self.device), y.to(self.device))
        self.test_class_auc(detection_preds.to(self.device), y.to(self.device))
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
        self.sess.close()
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


NoiseprintImageEvalPLWrapper = NoiseprintImageEvalPLWrapper(
    chkpt_folder,
    model_name,
    QF,
)
