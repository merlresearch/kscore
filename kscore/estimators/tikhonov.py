# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import math
from collections import namedtuple

import torch

from kscore.kernels import CurlFreeIMQ
from kscore.utils import conjugate_gradient, random_choice

from .base import BaseEstimator


class Tikhonov(BaseEstimator):
    def __init__(
        self,
        lam,
        kernel=None,
        truncated_tikhonov=False,
        subsample_rate=None,
        use_cg=True,
        tol_cg=1.0e-4,
        maxiter_cg=40,
        dtype=torch.float32,
    ):
        if kernel is None:
            kernel = CurlFreeIMQ()
        super().__init__(lam, kernel, dtype)
        self._use_cg = use_cg
        self._tol_cg = tol_cg
        self._subsample_rate = subsample_rate
        self._maxiter_cg = maxiter_cg
        self._truncated_tikhonov = truncated_tikhonov

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"lam={self._lam}, "
            + f"use_cg={self._use_cg}, "
            + f"tol_cg={self._tol_cg}, "
            + f"subsample_rate={self._subsample_rate}, "
            + f"maxiter_cg={self._maxiter_cg}, "
            + f"truncated_tikhonov={self._truncated_tikhonov}, "
            + f"dtype={self._dtype}, "
            + f"kernel={self._kernel}"
            + ")"
        )

    def fit(self, samples, kernel_hyperparams=None):
        if self._subsample_rate is None:
            return self._fit_exact(samples, kernel_hyperparams)
        else:
            return self._fit_subsample(samples, kernel_hyperparams)

    def _compute_energy(self, x):
        if self._subsample_rate is None and not self._truncated_tikhonov:
            Kxq, div_xq = self._kernel.kernel_energy(x, self._samples, kernel_hyperparams=self._kernel_hyperparams)
            div_xq = torch.mean(div_xq, dim=-1) / self._lam
            Kxq = Kxq.reshape([x.shape[-2], -1])
            energy = torch.matmul(Kxq, self._coeff)
            energy = energy.reshape(-1) - div_xq
        else:
            Kxq = self._kernel.kernel_energy(
                x, self._samples, kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False
            )
            Kxq = Kxq.reshape(x.shape[-2], -1)
            energy = -torch.matmul(Kxq, self._coeff)
            energy = energy.reshape(-1)
        return energy

    def compute_gradients(self, x):
        d = x.shape[-1]
        if self._subsample_rate is None and not self._truncated_tikhonov:
            Kxq_op, div_xq = self._kernel.kernel_operator(
                x, self._samples, kernel_hyperparams=self._kernel_hyperparams
            )
            div_xq = torch.mean(div_xq, dim=-2) / self._lam
            grads = Kxq_op.apply(self._coeff)
            grads = grads.reshape([-1, d]) - div_xq
        else:
            Kxq_op = self._kernel.kernel_operator(
                x, self._samples, kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False
            )
            grads = -Kxq_op.apply(self._coeff)
            grads = grads.reshape([-1, d])
        return grads

    def _fit_subsample(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams

        M = samples.shape[-2]
        N = math.floor(M * self._subsample_rate)
        d = samples.shape[-1]

        subsamples = random_choice(samples, N)
        Knn_op = self._kernel.kernel_operator(
            subsamples, subsamples, kernel_hyperparams=kernel_hyperparams, compute_divergence=False
        )
        Knm_op, K_div = self._kernel.kernel_operator(subsamples, samples, kernel_hyperparams=kernel_hyperparams)
        self._samples = subsamples

        if self._use_cg:

            def apply_kernel(v):
                return Knm_op.apply(Knm_op.apply_adjoint(v)) / M + self._lam * Knn_op.apply(v)

            linear_operator = namedtuple("LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])
            Kcg_op = linear_operator(
                shape=Knn_op.shape,
                dtype=Knn_op.dtype,
                apply=apply_kernel,
                apply_adjoint=apply_kernel,
            )
            H_dh = torch.mean(K_div, dim=-2)
            H_dh = H_dh.reshape(N * d)
            conj_ret = conjugate_gradient(Kcg_op, H_dh, max_iter=self._maxiter_cg, tol=self._tol_cg)
            self._coeff = conj_ret.x.reshape([N * d, 1])
        else:
            Knn = Knn_op.kernel_matrix(flatten=True)
            Knm = Knm_op.kernel_matrix(flatten=True)
            K_inner = torch.matmul(Knm, Knm.T) / M + self._lam * Knn
            H_dh = torch.mean(K_div, dim=-2)

            if self._kernel.kernel_type() == "diagonal":
                K_inner += 1.0e-7 * torch.eye(N, device=samples.device)
                H_dh = H_dh.reshape([N, d])
            else:
                # The original Nystrom KEF estimator (Sutherland et al., 2018).
                # Adding the small identity matrix is necessary for numerical stability.
                K_inner += 1.0e-7 * torch.eye(N * d, device=samples.device)
                H_dh = H_dh.reshape([N * d, 1])
            self._coeff = torch.linalg.solve(K_inner, H_dh).reshape([N * d, 1])

    def _fit_exact(self, samples, kernel_hyperparams=None):
        # samples: [M, d]
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = samples.shape[-2]
        d = samples.shape[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples, kernel_hyperparams=kernel_hyperparams)

        if self._use_cg:
            if self._truncated_tikhonov:

                def apply_kernel(v):
                    return K_op.apply(K_op.apply(v) / M + self._lam * v)

            else:

                def apply_kernel(v):
                    return K_op.apply(v) + M * self._lam * v

            linear_operator = namedtuple("LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])
            Kcg_op = linear_operator(
                shape=K_op.shape,
                dtype=K_op.dtype,
                apply=apply_kernel,
                apply_adjoint=apply_kernel,
            )
            H_dh = torch.mean(K_div, dim=-2)
            H_dh = H_dh.reshape([M * d]) / self._lam
            conj_ret = conjugate_gradient(Kcg_op, H_dh, max_iter=self._maxiter_cg, tol=self._tol_cg)
            self._coeff = conj_ret.x.reshape([M * d, 1])
        else:
            K = K_op.kernel_matrix(flatten=True)
            H_dh = torch.mean(K_div, dim=-2)
            if self._kernel.kernel_type() == "diagonal":
                identity = torch.eye(M, device=samples.device)
                H_shape = [M, d]
            else:
                identity = torch.eye(M * d, device=samples.device)
                H_shape = [M * d, 1]

            if self._truncated_tikhonov:
                # The Nystrom version of KEF with full samples.
                # See Example 3.6 for more details.
                K = torch.matmul(K, K) / M + self._lam * K + 1.0e-7 * identity
            else:
                # The original KEF estimator (Sriperumbudur et al., 2017).
                K += M * self._lam * identity
            H_dh = H_dh.reshape(H_shape) / self._lam
            self._coeff = torch.linalg.solve(K, H_dh).reshape([M * d, 1])
