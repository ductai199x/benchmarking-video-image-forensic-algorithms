import lightning.pytorch as pl
import torch
from torchmetrics import AUROC, Accuracy, F1Score, MatthewsCorrCoef

from .fsg import FSG
from .localization import PatchLocalization, pixel_loc_from_patch_pred
from .spectral_utils import eigap01, laplacian, spectral_cluster


def batch_fn(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


model_config = {
    "patch_size": (128, 128),
    "lr": 2e-5,
    "decay_step": 2,
    "decay_rate": 0.75,
    "num_pre_filters": 6,
    "input_dim": 200,
    "map1_dim": 2048,
    "map2_dim": 64,
}

state_dict_saved_path = "/media/nas2/trained_models_repository/fsg_pytorch/image/fsg_image_128_pytorch_from_tf1.pt"


class FSGWholeImageEvalPLWrapper(pl.LightningModule):
    def __init__(
        self,
        model_config,
        state_dict_saved_path,
        patch_size=128,
        overlap=64,
        **kwargs,
    ):
        super().__init__()
        self.model = FSG(**model_config)
        self.model.load_state_dict(torch.load(state_dict_saved_path))
        self.model.eval()

        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

        self.test_class_acc = Accuracy()
        self.test_class_auc = AUROC(num_classes=2, compute_on_step=False)
        self.test_loc_f1 = F1Score()
        self.test_loc_mcc = MatthewsCorrCoef(num_classes=2)

    def get_features(self, image_patches):
        patches_features = []
        for batch in batch_fn(image_patches, 256):
            batch = batch.float().to(self.device)
            feats = self.model.mislnet(batch).detach().cpu()
            patches_features.append(feats)
        patches_features = torch.vstack(patches_features)
        return patches_features

    def get_sim_scores(self, patch_pairs):
        patches_sim_scores = []
        for batch in batch_fn(patch_pairs, 512):
            batch = batch.permute(1, 0, 2).float().to(self.device)
            scores = self.model.comparenet(batch).detach().cpu()
            scores = torch.nn.functional.softmax(scores, dim=1)
            patches_sim_scores.append(scores)
        patches_sim_scores = torch.vstack(patches_sim_scores)
        return patches_sim_scores

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape

        # split images into batches of patches: B x C x H x W -> B x (NumPatchHeight x NumPatchWidth) x C x PatchSize x PatchSize
        batched_patches = (
            x.unfold(2, self.patch_size, self.stride)
            .unfold(3, self.patch_size, self.stride)
            .permute(0, 2, 3, 1, 4, 5)
        )
        batched_patches = batched_patches.contiguous().view(
            B, -1, C, self.patch_size, self.patch_size
        )
        B, P, C, P_H, P_W = batched_patches.shape

        # get the (x, y) coordinates of the top left of each patch in the image
        x_inds = torch.arange(W).unfold(0, self.patch_size, self.stride)[:, 0]
        y_inds = torch.arange(H).unfold(0, self.patch_size, self.stride)[:, 0]
        xy_inds = [(ii, jj) for jj in y_inds for ii in x_inds]

        # initialize batch predictions:
        detection_preds = []
        localization_preds = []

        # loop through the patches for each image in the batch
        for image_patches in batched_patches:
            patches_features = self.get_features(image_patches)
            patch_cart_prod = torch.cartesian_prod(torch.arange(P), torch.arange(P))
            patch_pairs = patches_features[patch_cart_prod]
            patches_sim_scores = self.get_sim_scores(patch_pairs)

            sim_mat = patches_sim_scores[:, 1].reshape(P, P)
            sim_mat = 0.5 * (sim_mat + sim_mat.T)
            sim_mat.fill_diagonal_(1.0)

            normL = laplacian(
                sim_mat, laplacian_type="sym"
            )  # normalized laplacian matrix
            normgap = eigap01(normL)  # normalized spectral gap
            prediction = spectral_cluster(normL)
            pat_loc = PatchLocalization(
                inds=xy_inds, patch_size=128, prediction=~prediction
            )
            # here we flip the label for easier visualization..
            # note the label=0 in the line above
            # and the ~pat_loc.prediction in the line below
            pix_loc = pixel_loc_from_patch_pred(
                prediction=~pat_loc.prediction,
                inds=xy_inds,
                patch_size=128,
                image_shape=(H, W),
                threshold=0.5,
                normalization=False,
            )
            pix_loc_pred = pix_loc.prediction / pix_loc.prediction.max()

            detection_preds.append(normgap)
            localization_preds.append(pix_loc_pred)

        self.test_class_acc(
            torch.tensor(detection_preds).to(self.device), y.to(self.device)
        )
        self.test_class_auc(
            torch.tensor(detection_preds).to(self.device), y.to(self.device)
        )
        for i in range(B):
            loc_pred = torch.tensor(localization_preds[i]).to(self.device)
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
        self.log("test_class_fnr", neg_preds.sum() / neg_labels.sum())


FSGWholeImageEvalPLWrapper = FSGWholeImageEvalPLWrapper(
    model_config,
    state_dict_saved_path,
    patch_size=128,
    overlap=64,
)
