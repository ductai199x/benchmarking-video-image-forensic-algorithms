#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 14:26:19 2019

@author: owen
"""
import numpy as np


def laplacian(S, laplacian_type=None, degree_axis=1):
    # based on A Tutorial on Spectral Clustering by Ulrike von Luxburg
    # https://arxiv.org/pdf/0711.0189.pdf

    A = np.copy(S)  # adjacency matrix
    np.fill_diagonal(A, 0)  # put zeros on diagonal of Adjacency mat

    D = np.diag(A.sum(axis=degree_axis))  # degree matrix

    L = D - A  # Laplacian matrix

    # Sec. 3.1 The unnormalized graph Laplacian
    if laplacian_type is None:
        return L

    # Sec. 3.2 The normalized graph Laplacians
    elif laplacian_type == "sym":
        # SYMMETRIC NORMALIZED LAPLACIAN
        Dsym = np.diag(A.sum(axis=degree_axis) ** -0.5)
        Lsym = np.matmul(np.matmul(Dsym, L), Dsym)
        return Lsym

    elif laplacian_type == "rw":
        # RANDOM WALK NORMALIZED LAPLACIAN
        Drw = np.diag(A.sum(axis=degree_axis) ** -1)
        Lrw = np.matmul(Drw, L)
        return Lrw


def get_eigs(L):
    eigs_ = np.linalg.eigvals(L)
    eigs_ = np.sort(np.absolute(eigs_))
    return eigs_


def eigap01(L):
    eigs_ = get_eigs(L)
    gap01 = eigs_[1] - eigs_[0]
    return gap01


def spectral_cluster(L):
    e, v = np.linalg.eig(L)
    inds_sorted = np.argsort(e)
    vec = v[
        :, inds_sorted[1]
    ]  # pull out eigenvector associated with first non-zero eigenvalue
    prediction = vec > 0
    return prediction
