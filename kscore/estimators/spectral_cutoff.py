# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import math

import torch

from kscore.kernels import CurlFreeIMQ

from .base import BaseEstimator


class SpectralCutoff(BaseEstimator):
    def __init__(self, lam=None, keep_rate=None, kernel=None, dtype=torch.float32):
        if kernel is None:
            kernel = CurlFreeIMQ()
        if lam is not None and keep_rate is not None:
            raise ValueError("Cannot specify `lam` and `keep_rate` simultaneously.")
        if lam is None and keep_rate is None:
            raise ValueError("Both `lam` and `keep_rate` are `None`.")
        super().__init__(lam, kernel, dtype)
        self._keep_rate = keep_rate

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"lam={self._lam}, "
            + f"keep_rate={self._keep_rate}, "
            + f"dtype={self._dtype}, "
            + f"kernel={self._kernel}"
            + ")"
        )

    def fit(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = samples.shape[-2]

        K_op, K_div = self._kernel.kernel_operator(samples, samples, kernel_hyperparams=kernel_hyperparams)
        K = K_op.kernel_matrix(flatten=True)

        eigen_values, eigen_vectors = torch.linalg.eigh(K / M)
        if self._keep_rate is not None:
            total_num = K.shape[0]
            n_eigen = math.floor(total_num * self._keep_rate)
        else:
            n_eigen = torch.sum(eigen_values > self._lam)
        n_eigen = max(n_eigen, 1)
        eigen_values = eigen_values[..., -n_eigen:]

        # [Md, eigens], or [M, eigens]
        eigen_vectors = eigen_vectors[..., -n_eigen:] / eigen_values

        H_dh = torch.mean(K_div, dim=-2)  # [M, d]
        if self._kernel.kernel_type() == "diagonal":
            truncated_Kinv = eigen_vectors.unsqueeze(-2) * eigen_vectors
            truncated_Kinv = torch.sum(truncated_Kinv, dim=-1)  # [M, M]
            self._coeff = torch.matmul(truncated_Kinv, H_dh)
        else:
            H_dh = H_dh.reshape([-1, 1])  # [Md, 1]
            self._coeff = torch.sum(eigen_vectors * torch.sum(eigen_vectors * H_dh, dim=0), dim=-1, keepdims=True)
        self._coeff /= torch.tensor(M, device=samples.device).type(self._dtype)

    def _compute_energy(self, x):
        Kxq = self._kernel.kernel_energy(
            x, self._samples, kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False
        )
        Kxq = Kxq.reshape([x.shape[-2], -1])
        energy = torch.matmul(Kxq, self._coeff)
        return -energy.reshape([-1])

    def compute_gradients(self, x):
        Kxq_op = self._kernel.kernel_operator(
            x, self._samples, kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False
        )
        if self._kernel.kernel_type() == "diagonal":
            Kxq = Kxq_op.kernel_matrix(flatten=True)
            grads = torch.matmul(Kxq, self._coeff)
        else:
            d = x.shape[-1]
            grads = Kxq_op.apply(self._coeff).reshape([-1, d])
        return -grads
