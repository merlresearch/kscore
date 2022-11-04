# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import math

import torch

from kscore.kernels import CurlFreeIMQ

from .base import BaseEstimator


class NuMethod(BaseEstimator):
    def __init__(self, lam=None, iternum=None, kernel=None, nu=1.0, dtype=torch.float32):
        if kernel is None:
            kernel = CurlFreeIMQ()
        if lam is not None and iternum is not None:
            raise ValueError("Cannot specify `lam` and `iternum` simultaneously.")
        if lam is None and iternum is None:
            raise ValueError("Both `lam` and `iternum` are `None`.")
        if iternum is not None:
            lam = 1.0 / iternum**2
        else:
            iternum = math.floor(1.0 / math.sqrt(lam)) + 1
        super().__init__(lam, kernel, dtype)
        self._nu = nu
        self._iternum = iternum

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"lam={self._lam}, "
            + f"iternum={self._iternum}, "
            + f"kernel={self._kernel}, "
            + f"nu={self._nu}, "
            + f"dtype={self._dtype}"
            + ")"
        )

    def fit(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = samples.shape[-2]
        d = samples.shape[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples, kernel_hyperparams=kernel_hyperparams)

        # H_dh: [Md, 1]
        H_dh = torch.mean(K_div, dim=-2).reshape([M * d, 1])

        def get_next(t, a, pa, c, pc):
            # nc <- c <- pc
            ft = t
            nu = self._nu
            u = (
                (ft - 1.0)
                * (2.0 * ft - 3.0)
                * (2.0 * ft + 2.0 * nu - 1.0)
                / ((ft + 2.0 * nu - 1.0) * (2.0 * ft + 4.0 * nu - 1.0) * (2.0 * ft + 2.0 * nu - 3.0))
            )
            w = (
                4.0
                * (2.0 * ft + 2.0 * nu - 1.0)
                * (ft + nu - 1.0)
                / ((ft + 2.0 * nu - 1.0) * (2.0 * ft + 4.0 * nu - 1.0))
            )
            nc = (1.0 + u) * c - w * (a * H_dh + K_op.apply(c)) / M - u * pc
            na = (1.0 + u) * a - u * pa - w
            return na, a, nc, c

        a = -(4.0 * self._nu + 2) / (4.0 * self._nu + 1)
        pa = 0.0
        c = torch.zeros_like(H_dh)
        pc = torch.zeros_like(H_dh)

        for t in range(2, self._iternum + 1):
            a, pa, c, pc = get_next(t, a, pa, c, pc)

        self._coeff = (a, c)

    def _compute_energy(self, x):
        Kxq, div_xq = self._kernel.kernel_energy(x, self._samples, kernel_hyperparams=self._kernel_hyperparams)
        Kxq = Kxq.reshape([x.shape[-2], -1])
        div_xq = torch.mean(div_xq, dim=-1) * self._coeff[0]
        energy = torch.matmul(Kxq, self._coeff[1]).reshape([-1]) + div_xq
        return energy

    def compute_gradients(self, x):
        d = x.shape[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(x, self._samples, kernel_hyperparams=self._kernel_hyperparams)
        div_xq = torch.mean(div_xq, dim=-2) * self._coeff[0]
        grads = Kxq_op.apply(self._coeff[1])
        grads = grads.reshape([-1, d]) + div_xq
        return grads
