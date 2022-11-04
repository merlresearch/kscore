# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import torch

from kscore.utils import median_heuristic

from .square_curlfree import SquareCurlFree


class CurlFreeGaussian(SquareCurlFree):
    def __init__(self, heuristic_hyperparams=median_heuristic):
        super().__init__(heuristic_hyperparams)

    def __repr__(self):
        return f"{type(self).__name__}(heuristic_hyperparams={self._heuristic_hyperparams})"

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        r"""
        Construct the curl-free kernel $-\nabla^2 \psi(\|x - y\|^2)$.
        You need to provide the first, second and third derivatives of $\psi$.
        See eq. (21) and eq. (22).
        """
        inv_sqr_sigma = 0.5 / torch.square(torch.tensor(sigma, device=r.device))
        rbf = torch.exp(-norm_rr * inv_sqr_sigma)
        G_1st = -rbf * inv_sqr_sigma
        G_2nd = -G_1st * inv_sqr_sigma
        G_3rd = -G_2nd * inv_sqr_sigma
        return r, norm_rr, G_1st, G_2nd, G_3rd
