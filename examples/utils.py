# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import matplotlib.pyplot as plt
import numpy as np
import torch

import kscore


def get_estimator(args):
    kernel_dicts = {
        "diagonal_adaptive_gaussian": kscore.kernels.DiagonalAdaptiveGaussian,
        "curlfree_adaptive_gaussian": kscore.kernels.CurlFreeAdaptiveGaussian,
        "curlfree_imq": kscore.kernels.CurlFreeIMQ,
        "curlfree_rbf": kscore.kernels.CurlFreeGaussian,
        "diagonal_imq": kscore.kernels.DiagonalIMQ,
        "diagonal_rbf": kscore.kernels.DiagonalGaussian,
    }

    estimator_dicts = {
        "tikhonov": kscore.estimators.Tikhonov,
        "nu": kscore.estimators.NuMethod,
        "landweber": kscore.estimators.Landweber,
        "spectral_cutoff": kscore.estimators.SpectralCutoff,
        "stein": kscore.estimators.Stein,
    }

    if args.estimator == "tikhonov_nystrom":
        estimator = kscore.estimators.Tikhonov(
            lam=args.lam, subsample_rate=args.subsample_rate, kernel=kernel_dicts[args.kernel]()
        )
    else:
        estimator = estimator_dicts[args.estimator](lam=args.lam, kernel=kernel_dicts[args.kernel]())
    return estimator


def add_estimator_params(parser):
    parser.add_argument(
        "--estimator",
        type=str,
        default="nu",
        help="score estimator.",
        choices=["nu", "landweber", "tikhonov", "tikhonov_nystrom", "spectral_cutoff", "stein"],
    )
    parser.add_argument("--lam", type=float, default=1.0e-5, help="regularization parameter.")
    parser.add_argument(
        "--kernel",
        type=str,
        default="curlfree_imq",
        help="matrix-valued kernel.",
        choices=[
            "curlfree_imq",
            "curlfree_rbf",
            "diagonal_imq",
            "diagonal_rbf",
            "diagonal_adaptive_gaussian",
            "curlfree_adaptive_gaussian",
        ],
    )
    parser.add_argument(
        "--subsample_rate", default=None, type=float, help="subsample rate used in the Nystrom approimation."
    )
    return parser


def plot_vector_field(X, Y, normalize=False):
    if normalize:
        for i in range(Y.shape[0]):
            norm = (Y[i][0] ** 2 + Y[i][1] ** 2) ** 0.5
            Y[i] /= norm
    plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1])


def linspace_2d(size, lower_box, upper_box):
    xs_1d = np.linspace(lower_box, upper_box, size)
    xs = []
    for i in xs_1d:
        for j in xs_1d:
            xs.append([i, j])
    xs = torch.tensor(xs, dtype=torch.float32)
    return xs


def clip_energy(energy, threshold=24):
    max_v = torch.max(energy)
    clip_v = max_v - threshold
    return torch.maximum(energy, clip_v) - max_v
