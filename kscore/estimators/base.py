# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
class BaseEstimator:
    def __init__(self, lam, kernel, dtype):
        self._lam = lam
        self._kernel = kernel
        self._coeff = None
        self._kernel_hyperparams = None
        self._samples = None
        self._dtype = dtype

    def fit(self, samples, kernel_hyperparams):  # pragma: no cover
        raise NotImplementedError("Not implemented score estimator!")

    def compute_gradients(self, x):  # pragma: no cover
        raise NotImplementedError("Not implemented score estimator!")

    def compute_energy(self, x):  # pragma: no cover
        if self._kernel.kernel_type() != "curl-free":
            raise RuntimeError("Only curl-free kernels have well-defined energy.")
        return self._compute_energy(x)

    def _compute_energy(self, x):  # pragma: no cover
        raise NotImplementedError("Not implemented score estimator!")
