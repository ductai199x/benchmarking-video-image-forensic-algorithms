#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:42:58 2019

@author: owen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import pickle


class PatchLocalization:
    # has:   inds
    #       prediction
    #       groundtruth
    #       patch_size

    # methods:
    #       plot_heatmap
    #       score_patch_level

    def __init__(self, inds, patch_size, prediction):
        self.prediction = prediction
        self.inds = inds
        self.patch_size = patch_size

    def plot_heatmap(self, image, label=1, alpha=0.25, color="r"):

        fig, ax = plot_patch_heatmap(
            self.inds,
            self.prediction,
            self.patch_size,
            image,
            label=label,
            alpha=alpha,
            color=color,
        )
        return fig, ax

    def pixel_mask(self, label, mask_dims=None, threshold=None):

        mask = pixel_mask(
            self.inds,
            self.prediction,
            self.patch_size,
            label=label,
            mask_dims=mask_dims,
            threshold=threshold,
        )

        return mask

    def score_f1(
        self, ground_truth_mask, upper_pct=75, lower_pct=25, invert_mask=False
    ):

        if invert_mask:
            ground_truth_mask = np.abs(1 - ground_truth_mask)

        y = patch_score_from_mask(
            ground_truth_mask,
            self.inds,
            self.patch_size,
            upper_pct=upper_pct,
            lower_pct=lower_pct,
        )

        #        p = np.zeros(self.prediction.shape)
        #        p[self.prediction==label] = 1

        precision, recall, f1 = score_f1(y, self.prediction)

        return precision, recall, f1

    def score_mcc(
        self, ground_truth_mask, upper_pct=75, lower_pct=25, invert_mask=False
    ):

        if invert_mask:
            ground_truth_mask = np.abs(1 - ground_truth_mask)

        y = patch_score_from_mask(
            ground_truth_mask,
            self.inds,
            self.patch_size,
            upper_pct=upper_pct,
            lower_pct=lower_pct,
        )

        #
        #        p = np.zeros(self.prediction.shape)
        #        p[self.prediction==label] = 1

        mcc = score_mcc(y, self.prediction)

        return mcc


class dbScorer:
    def __init__(self, list_of_pmap_mask_tuples, info_dict={}):

        for tup in list_of_pmap_mask_tuples:
            pmap = tup[0]
            mask = tup[1]
            if not pmap.shape == mask.shape:
                raise Exception("Prediction Map and Mask are not Same dimensions")

        self.tuples = list_of_pmap_mask_tuples
        self.info = info_dict

    def score_best_thresh_per_image(
        self, score_type="mcc", thresh=np.linspace(0, 1, 11), quiet=False
    ):
        if score_type == "f1":
            score_func = lambda m, p: score_f1(m, p)[2]
        elif score_type == "mcc":
            score_func = lambda m, p: score_mcc(m, p)

        # score entire database by selecting the best threshold, per image
        # we can use a fixed threshold by setting thresh=[fixed_thresh_value]

        # score_func inputs two values: m (image mask, where 1 = forged, and 0=unaltered),
        # and p (predictions, where highervalue indicates forged and lower value indicates unaltered)
        # m and p must be same size

        score_list = []
        for tup in tqdm(self.tuples, desc="Scoring DB", disable=quiet):
            p = tup[0]
            m = tup[1]

            # for each image iterate through all thresholds
            max_score = 0  # worst score an image can get is 0. this occasionally happens when no patches are detected
            for tt in thresh:
                # try with predictions, and inverted predictions
                score1 = score_func(m, p > tt)
                score2 = score_func(m, p <= tt)
                max_score = np.nanmax((max_score, score1, score2))  # pick the max

            score_list.append(max_score)  # aggregate best score for each image
        return np.mean(score_list), score_list  # report mean, and list of scores

    def score_best_thresh_for_db_imavg(
        self, score_type="mcc", thresh=np.linspace(0, 1, 21), quiet=False
    ):
        db_score_list = []
        for tt in tqdm(thresh, desc="Scoring DB", disable=quiet):
            score, _ = self.score_best_thresh_per_image(
                score_type, thresh=[tt], quiet=True
            )
            db_score_list.append(score)

        best_score = np.max(db_score_list)
        best_thresh = thresh[np.argmax(db_score_list)]

        # return best score, associated threshold, and all scores with associated thresholds
        return best_score, best_thresh, list(zip(thresh, db_score_list))

    def score_thresh_pix_total(self, score_type="mcc", thresh=0.5):

        if score_type == "f1":
            score_func = lambda m, p: score_f1(m, p)[2]
        elif score_type == "mcc":
            score_func = lambda m, p: score_mcc(m, p)

        # aggregate all predictions (p) and masks (m) from db
        p_list = []
        m_list = []
        for tup in self.tuples:
            p = tup[0].ravel()
            m = tup[1].ravel()

            score1 = score_func(m, p > thresh)
            score2 = score_func(m, p <= thresh)

            # see which way the thresholding is greater
            if (score1 > score2) | (~np.isnan(score1) & np.isnan(score2)):
                p_list.append(p > thresh)
            elif (score1 <= score2) | (~np.isnan(score2) & np.isnan(score1)):
                p_list.append(p <= thresh)
            else:
                p_list.append(p > thresh)

            m_list.append(m)

        p_all = np.hstack(p_list)
        m_all = np.hstack(m_list)

        return score_func(m_all, p_all)

    def score_best_thresh_pix_total(
        self, score_type, thresh=np.linspace(0, 1, 21), quiet=False
    ):
        maxscore = 0
        bestthresh = None

        for tt in tqdm(thresh, desc="Scoring DB", disable=quiet):
            score = self.score_thresh_pix_total(score_type, tt)

            if score > maxscore:
                maxscore = score
                bestthresh = tt

        return maxscore, bestthresh

    def score_auc_pix_total(self, T=np.linspace(0, 1, 101)):
        list_v0 = []
        list_v1 = []
        for tup in tqdm(self.tuples):

            m = tup[1]  # mask
            p = tup[0]  # prediction

            if np.min(p) < 0 or np.max(p) > 1:
                raise Exception("Roc requires predictions between 0 and 1")

            score = score_mcc(m, p > 0.5)

            if score > 0:  # predictions don't need to be flipped
                v0 = p[m == 0]  # prediction values for authentic regions
                v1 = p[m == 1]  # prediction values for forged regions

            else:  # flip predictions
                v0 = 1 - p[m == 0]  # prediction values for authentic regions
                v1 = 1 - p[m == 1]  # prediction values for forged regions

            list_v0.append(v0)
            list_v1.append(v1)

        v0 = np.hstack(list_v0)
        v1 = np.hstack(list_v1)
        pfa, pd = roc(v0, v1, T)
        auc = np.trapz(np.flip(pd), np.flip(pfa))

        return auc, pd, pfa

    #    def score_auc(self)

    #    def score_db_pixel_total():

    def save(self, name):
        with open(name, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def scorer_from_file(filename):
    with open(filename, "rb") as f:
        scorer = pickle.load(f)
    return scorer


class PixelLocalization:
    def __init__(self, prediction_map):
        self.prediction = prediction_map

    def plot(self, image, cmap=plt.cm.autumn_r, max_alpha=0.8):
        f, ax = plot_pixel_heatmap(self.prediction, image, cmap, max_alpha)
        return f, ax

    def score_f1(self, ground_truth_mask, invert_mask=False):

        if invert_mask:
            ground_truth_mask = np.abs(1 - ground_truth_mask)

        precision, recall, f1 = score_f1(ground_truth_mask, self.prediction)

        return precision, recall, f1

    def score_mcc(self, ground_truth_mask, invert_mask=False):

        if invert_mask:
            ground_truth_mask = np.abs(1 - ground_truth_mask)

        mcc = score_mcc(ground_truth_mask, self.prediction)

        return mcc


def pixel_loc_from_patch_pred(
    prediction,
    inds,
    patch_size,
    image_shape,
    threshold,
    smoothing="Gaussian",
    smoothing_window=32,
    normalization=True,
):
    # inds = sg.inds
    # patch_size=sg.patch_size
    # image_shape = M.shape
    # smoothing = 'Gaussian'
    # smoothing_window = 32
    # normalization = True
    # threshold = 0.33

    # create patch localization from patch predictions
    patchloc = PatchLocalization(inds, patch_size, prediction)

    # output pixel representation
    pixel_prediction = patchloc.pixel_mask(
        label=1, mask_dims=image_shape, threshold=None
    )

    # smoothing
    if smoothing == "Gaussian":
        pixel_prediction = gaussian_filter(
            pixel_prediction, (smoothing_window, smoothing_window), mode="reflect"
        )

    # normalization
    if normalization == True:
        coverage = patch_coverage(inds, patch_size, image_shape)

        if smoothing == "Gaussian":
            # this needs to be the same as smooth_pixel_prediction
            coverage = gaussian_filter(
                coverage, (smoothing_window, smoothing_window), mode="reflect"
            )

        # inversely weight predictions with less coverage
        label_map = pixel_prediction / coverage > threshold

    else:
        label_map = pixel_prediction > threshold

    pixloc = PixelLocalization(label_map)

    return pixloc


def pixel_pred_from_patch_pred(
    prediction,
    inds,
    patch_size,
    image_shape,
    smoothing="Gaussian",
    smoothing_window=32,
    normalization=True,
):
    # inds = sg.inds
    # patch_size=sg.patch_size
    # image_shape = M.shape
    # smoothing = 'Gaussian'
    # smoothing_window = 32
    # normalization = True
    # threshold = 0.33

    # create patch localization from patch predictions
    patchloc = PatchLocalization(inds, patch_size, prediction)

    # output pixel representation
    pixel_prediction = patchloc.pixel_mask(
        label=1, mask_dims=image_shape, threshold=None
    )

    # smoothing
    if smoothing == "Gaussian":
        pixel_prediction = gaussian_filter(
            pixel_prediction, (smoothing_window, smoothing_window), mode="reflect"
        )

    # normalization
    if normalization == True:
        coverage = patch_coverage(inds, patch_size, image_shape)

        if smoothing == "Gaussian":
            # this needs to be the same as smooth_pixel_prediction
            coverage = gaussian_filter(
                coverage, (smoothing_window, smoothing_window), mode="reflect"
            )

        # inversely weight predictions with less coverage
        pixel_prediction = pixel_prediction / coverage

    return pixel_prediction


def plot_pixel_heatmap(prediction_map, image, cmap=plt.cm.autumn_r, max_alpha=1):

    f, ax = plt.subplots(1)
    ax.set_adjustable("box")
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, max_alpha, cmap.N + 3)
    #    mycmap._lut[:,-1] = np.logspace(0, 0.25, cmap.N+3)-1

    ax.imshow(prediction_map, cmap=mycmap)

    return f, ax


def plot_patch_heatmap(
    inds, prediction, patch_size, image, label=1, alpha=0.25, color="r"
):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    for ii, _ in enumerate(inds):
        if (prediction[ii] == label) | (np.isnan(prediction[ii]) & np.isnan(label)):
            ax.add_patch(
                patches.Rectangle(
                    inds[ii],
                    patch_size,
                    patch_size,
                    facecolor=color,
                    alpha=alpha,
                    linewidth=0,
                )
            )
    return fig, ax


def pixel_mask(inds, prediction, patch_size, label, mask_dims=None, threshold=None):

    if mask_dims is None:
        max_inds = np.flip(np.max(np.array(inds), axis=0), axis=0)
        mask_dims = max_inds + patch_size

    mask = np.zeros(mask_dims, dtype=np.uint8)

    for ii, ind in enumerate(inds):
        if (prediction[ii] == label) | (np.isnan(prediction[ii]) & np.isnan(label)):
            xx = np.arange(ind[0], ind[0] + patch_size)
            yy = np.arange(ind[1], ind[1] + patch_size)
            mask[np.ix_(yy, xx)] = mask[np.ix_(yy, xx)] + 1

    if threshold is not None:
        mask = mask > threshold
    return mask


def compare_pixels_to_mask(
    inds, patch_prediction, patch_size, mask_dims, threshold, label=1, normalize=True
):

    # convert patch scores to pixel map, each pixel is the sum of patches that detect it
    patch_prediction_map = pixel_mask(
        inds, patch_prediction, patch_size, label=label, mask_dims=mask_dims
    )

    if normalize:  # determine detections based on relative maximum score for each pixel
        # map that has maximum score for each pixel, based on how many patches cover it
        pixel_coverage_map = patch_coverage(inds, patch_size, mask_dims)
        pixel_prediction_map = patch_prediction_map > (threshold * pixel_coverage_map)

    else:
        pixel_prediction_map = patch_prediction_map > threshold
    return pixel_prediction_map


def patch_score_from_mask(mask, inds, patch_size, upper_pct=50, lower_pct=50):
    upper_thresh = (upper_pct / 100) * (patch_size**2)
    lower_thresh = (lower_pct / 100) * (patch_size**2)

    list_y = []
    for ii, ind in enumerate(inds):
        xx = np.arange(ind[0], ind[0] + patch_size)
        yy = np.arange(ind[1], ind[1] + patch_size)
        m = mask[np.ix_(yy, xx)]

        if np.sum(m == True) >= upper_thresh:
            list_y.append(True)
        elif np.sum(m == False) > (patch_size**2) - lower_thresh:
            list_y.append(False)
        else:
            list_y.append(np.nan)

        y = np.array(list_y)

    return y


def patch_coverage(inds, patch_size, mask_dims, scale=None):
    coverage = pixel_mask(
        inds,
        np.zeros(len(inds)),
        patch_size,
        label=0,
        mask_dims=mask_dims,
        threshold=None,
    )
    if scale is not None:
        coverage = coverage * scale
    return coverage


def load_misl_mask(mask_path):
    M = plt.imread(mask_path)
    M = M[:, :, 0]  # get just one channel
    M = M.astype(int)  # convert to int array
    M = np.abs(1 - M)
    return M


def load_carvalho_mask(mask_path):
    M = plt.imread(mask_path)
    M = M[:, :, 0]  # get just one channel
    M = M.astype(int)  # convert to int array
    M = np.abs(1 - M)
    return M


def load_columbia_mask(mask_path):
    M_IN = plt.imread(mask_path)

    # intialize with nan -> no decision
    M = np.empty((M_IN.shape[0], M_IN.shape[1]))
    M[:] = np.nan

    pos_inds = M_IN[:, :, 1] >= 128  # green is forged area
    M[pos_inds] = 1

    neg_inds = M_IN[:, :, 0] >= 128  # red is unaltered area
    M[neg_inds] = 0

    return M


def load_korus_mask(mask_path, semantic_only=False):
    M_IN = plt.imread(mask_path)

    if len(M_IN.shape) > 2:  # if the mask has color / alpha channel
        M_IN = M_IN[:, :, 0]  # convert to 2D

    if semantic_only:
        M = M_IN > 0.75
    else:
        M = M_IN > 0.25

    M = M.astype(int)  # convert to int array
    return M


def score_f1(y, predictions):

    tp = np.sum(predictions[y == 1] == 1)
    fp = np.sum(predictions[y == 0] == 1)
    fn = np.sum(predictions[y == 1] == 0)

    precision = tp / (tp + fp)

    if (tp == 0) & (fn == 0):
        return precision, np.nan, np.nan

    recall = tp / (tp + fn)

    if (precision == 0) & (precision == 0):
        return precision, recall, np.nan

    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def score_mcc(y, predictions):

    tp = np.sum(predictions[y == 1] == 1).astype(float)
    tn = np.sum(predictions[y == 0] == 0).astype(float)
    fp = np.sum(predictions[y == 0] == 1).astype(float)
    fn = np.sum(predictions[y == 1] == 0).astype(float)

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if mcc_denominator == 0:
        return np.nan

    mcc = mcc_numerator / mcc_denominator

    return mcc


def roc(v0, v1, T=None, quiet=False):

    if T is None:
        # thresholds (calc at each auth and spliced metric point)
        T = np.sort(np.unique(np.concatenate((v0, v1))))

    pfa = []
    pd = []
    for t in tqdm(T, desc="Calculating ROC", disable=quiet):
        fa = np.sum(v0 >= t) / float(len(v0))  # false alarms at t
        pfa.append(fa)
        d = np.sum(v1 >= t) / float(len(v1))  # detections at t
        pd.append(d)
    vpfa = np.array(pfa)
    vpd = np.array(pd)

    return vpfa, vpd


def auc(mask, predictions, T=None):

    if np.min(predictions) < 0 or np.max(predictions) > 1:
        raise Exception("Roc requires predictions between 0 and 1")

    v0 = predictions[mask == 0]
    v1 = predictions[mask == 1]

    pfa1, pd1 = roc(v0, v1, T=T)
    pfa2, pd2 = roc(1 - v0, 1 - v1, T=T)

    auc1 = np.trapz(np.flip(pd1), np.flip(pfa1))
    auc2 = np.trapz(np.flip(pd2), np.flip(pfa2))

    if auc2 > auc1:
        return auc2, 1 - v0, 1 - v1
    elif auc1 > auc2:
        return auc1, v0, v1


#    return
