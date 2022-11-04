# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
from collections import namedtuple

import torch

from .base import BaseKernel


class Diagonal(BaseKernel):
    def __init__(self, heuristic_hyperparams):
        super().__init__("diagonal", heuristic_hyperparams)

    def _gram(self, x, y, kernel_hyperparams):
        return self._gram_impl(x, y, kernel_hyperparams)

    def _gram_impl(self, x, y, kernel_hyperparams):
        raise NotImplementedError("Gram matrix not implemented!")

    def kernel_operator(self, x, y, kernel_hyperparams, compute_divergence=True):
        d, M, N = x.shape[-1], x.shape[-2], y.shape[-2]
        K, divergence = self._gram(x, y, kernel_hyperparams)

        def kernel_op(z):
            # z: [N * d, L]
            L = z.shape[-1]
            z = z.reshape(N, d * L)
            ret = torch.matmul(K, z)
            return ret.reshape([M * d, L])

        def kernel_adjoint_op(z):
            # z: [M * d, L]
            L = z.shape[-1]
            z = z.reshape([M, d * L])
            ret = torch.matmul(K.T, z)
            return torch.reshape(ret, [N * d, L])

        def kernel_mat(flatten):
            if flatten:
                return K
            return K.unsqueeze(-1).unsqueeze(-1) * torch.eye(d, device=x.device)

        linear_operator = namedtuple("Operator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_adjoint=kernel_adjoint_op,
            kernel_matrix=kernel_mat,
        )

        if compute_divergence:
            return op, divergence
        return op
