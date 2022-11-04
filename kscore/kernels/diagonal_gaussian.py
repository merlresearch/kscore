# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import torch

from kscore.utils import median_heuristic

from .diagonal import Diagonal


class DiagonalGaussian(Diagonal):
    def __init__(self, heuristic_hyperparams=median_heuristic):
        super().__init__(heuristic_hyperparams)

    def __repr__(self):
        return f"{type(self).__name__}(heuristic_hyperparams={self._heuristic_hyperparams})"

    def _gram_impl(self, x, y, kernel_width):
        x_m = x.unsqueeze(-2)  # [M, 1, d]
        y_m = y.unsqueeze(-3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = torch.sum(diff * diff, dim=-1)  # [M, N]
        rbf = torch.exp(-0.5 * dist2 / kernel_width**2)  # [M, N]
        divergence = rbf.unsqueeze(-1) * (diff / kernel_width**2)
        return rbf, divergence
