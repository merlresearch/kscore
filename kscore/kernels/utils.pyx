# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2007-2021 The scikit-learn developers.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from https://github.com/scikit-learn/scikit-learn/blob/0.24.2/sklearn/manifold/_utils.pyx
# cython: boundscheck=False

cimport cython
from libc cimport math

import numpy as np

cimport numpy as np

np.import_array()


cdef extern from "numpy/npy_math.h":
    float NPY_INFINITY


cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5

cpdef get_tsne_betas(np.ndarray[np.float32_t, ndim=2] sqdistances, float desired_perplexity):
    """Binary search for sigmas of conditional Gaussians.

    This approximation reduces the computational complexity from O(N^2) to
    O(uN).

    Parameters
    ----------
    sqdistances : np.ndarray, shape (n_samples, n_neighbors)
        Distances between training samples and their k nearest neighbors.
        When using the exact method, this is a square (n_samples, n_samples)
        distance matrix. The TSNE default metric is "euclidean" which is
        interpreted as squared euclidean distance.

    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.

    Returns
    -------
    betas: array, shape (n_samples) - beta = 1 / 2 * sigma**2 for RBF kernel
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_samples = sqdistances.shape[0]
    cdef long n_neighbors = sqdistances.shape[1]
    cdef int using_neighbors = n_neighbors < n_samples
    # Precisions of conditional Gaussian distributions
    cdef float beta
    cdef float beta_min
    cdef float beta_max
    cdef float beta_sum = 0.0

    # Use log scale
    cdef float desired_entropy = math.log(desired_perplexity)
    cdef float entropy_diff

    cdef float entropy
    cdef float sum_Pi
    cdef float sum_disti_Pi
    cdef long i, j, k

    # This array is later used as a 32bit array. It has multiple intermediate
    # floating point additions that benefit from the extra precision
    cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros((n_samples, n_neighbors), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] betas = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    if n_samples == 1:
        # We are either comparing a single point to itself, or 1 point to 1 other point.
        if sqdistances[0, 0] == 0:
            betas[0, 0] = 1.0
            return betas
        else:
            betas[0, 0] = 1 / sqdistances[0,0]
            return betas
    elif n_neighbors == 1:
        # We have a collection of training points, and only 1 test point
        # Use the median heuristic; set sigma = median
        median_distance = np.median(np.sqrt(sqdistances))
        median_beta = 1 / median_distance
        for i in range(n_samples):
            betas[i, 0] = median_beta
        return betas

    for i in range(n_samples):
        beta_min = -NPY_INFINITY
        beta_max = NPY_INFINITY
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        for k in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed just over the nearest neighbors or over all data
            # if we're not using neighbors
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL

            sum_disti_Pi = 0.0
            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]
            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == NPY_INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -NPY_INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta
        for j in range(n_neighbors):
            betas[i, j] = beta

    # Symmetrize the betas for each (i, j) pair
    # Iterate over lower triangular
    for i in range(n_samples):
        for j in range(min(n_neighbors, i)):
            tmp = (betas[i, j] + betas[j, i]) / 2
            betas[i, j] = betas[j, i] = tmp

    return betas
