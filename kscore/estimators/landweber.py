# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import math

import torch

from kscore.kernels import CurlFreeIMQ

from .base import BaseEstimator


class Landweber(BaseEstimator):
    def __init__(self, lam=None, iternum=None, kernel=None, stepsize=1.0e-2, dtype=torch.float32):
        if kernel is None:
            kernel = CurlFreeIMQ()
        if lam is not None and iternum is not None:
            raise ValueError("Cannot specify `lam` and `iternum` simultaneously.")
        if lam is None and iternum is None:
            raise ValueError("Both `lam` and `iternum` are `None`.")
        if iternum is not None:
            lam = 1.0 / iternum
        else:
            iternum = math.floor(1.0 / lam) + 1
        super().__init__(lam, kernel, dtype)
        self._stepsize = stepsize
        self._iternum = iternum

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"lam={self._lam}, "
            + f"iternum={self._iternum}, "
            + f"kernel={self._kernel}, "
            + f"stepsize={self._stepsize}, "
            + f"dtype={self._dtype}"
            + ")"
        )

    def fit(self, samples, kernel_hyperparams=None):
        # samples: [M, d]
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = samples.shape[-2]
        d = samples.shape[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples, kernel_hyperparams=kernel_hyperparams)

        # H_dh: [Md, 1]
        H_dh = torch.mean(K_div, dim=-2).reshape(M * d, 1)

        def get_next(t, c):
            nc = c - self._stepsize * K_op.apply(c) - t * self._stepsize**2 * H_dh
            return nc

        coeff = torch.zeros_like(H_dh)
        for t in range(1, self._iternum):
            coeff = get_next(t, coeff)

        self._coeff = (-t * self._stepsize, coeff)

    def _compute_energy(self, x):
        Kxq, div_xq = self._kernel.kernel_energy(x, self._samples, kernel_hyperparams=self._kernel_hyperparams)
        Kxq = Kxq.reshape(x.shape[-2], -1)
        div_xq = torch.mean(div_xq, dim=-1) * self._coeff[0]
        energy = torch.matmul(Kxq, self._coeff[1]).reshape(-1) + div_xq
        return energy

    def compute_gradients(self, x):
        d = x.shape[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(x, self._samples, kernel_hyperparams=self._kernel_hyperparams)
        div_xq = torch.mean(div_xq, dim=-2) * self._coeff[0]
        grads = Kxq_op.apply(self._coeff[1])
        grads = grads.reshape(-1, d) + div_xq
        return grads
