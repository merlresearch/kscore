# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import torch

from kscore.utils import median_heuristic

from .square_curlfree import SquareCurlFree


class CurlFreeIMQ(SquareCurlFree):
    def __init__(self, heuristic_hyperparams=median_heuristic):
        super().__init__(heuristic_hyperparams)

    def __repr__(self):
        return f"{type(self).__name__}(heuristic_hyperparams={self._heuristic_hyperparams})"

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / torch.square(torch.tensor(sigma, device=r.device))
        imq = torch.rsqrt(1.0 + norm_rr * inv_sqr_sigma)  # [M, N]
        imq_2 = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        G_1st = -0.5 * imq_2 * inv_sqr_sigma * imq
        G_2nd = -1.5 * imq_2 * inv_sqr_sigma * G_1st
        G_3rd = -2.5 * imq_2 * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd


class CurlFreeIMQp(SquareCurlFree):
    def __init__(self, p=0.5, heuristic_hyperparams=median_heuristic):
        super().__init__(heuristic_hyperparams)
        self._p = p

    def __repr__(self):
        return f"{type(self).__name__}(heuristic_hyperparams={self._heuristic_hyperparams})"

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / torch.square(torch.tensor(sigma, device=r.device))
        imq = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        imq_p = torch.pow(imq, self._p)  # [M, N]
        G_1st = -(0.0 + self._p) * imq * inv_sqr_sigma * imq_p
        G_2nd = -(1.0 + self._p) * imq * inv_sqr_sigma * G_1st
        G_3rd = -(2.0 + self._p) * imq * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd
