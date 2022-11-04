# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import torch

from kscore.kernels import DiagonalIMQ

from .base import BaseEstimator


class Stein(BaseEstimator):
    def __init__(self, lam, kernel=None, dtype=torch.float32):
        # TODO: Implement curl-free kernels
        if kernel is None:
            kernel = DiagonalIMQ()
        if kernel.kernel_type() != "diagonal":  # pragma: no cover
            raise NotImplementedError("Only support diagonal kernels.")
        super().__init__(lam, kernel, dtype)

    def __repr__(self):
        return f"{type(self).__name__}(lam={self._lam}, dtype={self._dtype}, kernel={self._kernel})"

    def fit(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = samples.shape[-2]
        K_op, K_div = self._kernel.kernel_operator(samples, samples, kernel_hyperparams=kernel_hyperparams)
        K = K_op.kernel_matrix(flatten=True)
        # In the Stein estimator (Li & Turner, 2018), the regularization parameter is divided
        # by $M^2$, for the unified meaning of $\lambda$, we multiply this back.
        Mlam = M**2 * self._lam
        Kinv = torch.linalg.inv(K + Mlam * torch.eye(M, device=samples.device))
        H_dh = torch.sum(K_div, dim=-2)
        grads = -torch.matmul(Kinv, H_dh)
        self._coeff = {"Kinv": Kinv, "grads": grads, "Mlam": Mlam}

    def _compute_gradients_one(self, x):
        # Section 3.4 in Li & Turner (2018), the out-of-sample extension.
        Kxx = self._kernel.kernel_matrix(x, x, kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
        Kqx, Kqx_div = self._kernel.kernel_matrix(self._samples, x, kernel_hyperparams=self._kernel_hyperparams)
        KxqKinv = torch.matmul(Kqx.T, self._coeff["Kinv"])
        term1 = -1.0 / (Kxx + self._coeff["Mlam"] - torch.matmul(KxqKinv, Kqx))
        term2 = torch.matmul(Kqx.T, self._coeff["grads"]) - torch.matmul(KxqKinv + 1.0, Kqx_div.squeeze(-2))
        return torch.matmul(term1, term2)

    def compute_gradients(self, x):
        if x is self._samples:
            return self._coeff["grads"]
        else:

            def stein_dlog(y):
                stein_dlog_qx = self._compute_gradients_one(y.unsqueeze(0))
                stein_dlog_qx = stein_dlog_qx.squeeze(axis=-2)
                return stein_dlog_qx

            results = []
            for item in x:
                results.append(stein_dlog(item))

            return torch.stack(results)
