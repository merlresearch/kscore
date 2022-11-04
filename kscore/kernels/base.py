# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
class BaseKernel:
    def __init__(self, kernel_type, heuristic_hyperparams):
        self._kernel_type = kernel_type
        self._heuristic_hyperparams = heuristic_hyperparams

    def kernel_type(self):
        return self._kernel_type

    def heuristic_hyperparams(self, x, y):
        return self._heuristic_hyperparams(x, y)

    def kernel_operator(self, x, y, kernel_hyperparams, **kwargs):  # pragma: no cover
        raise NotImplementedError()

    def kernel_matrix(self, x, y, kernel_hyperparams=None, flatten=True, compute_divergence=True):
        if compute_divergence:
            op, divergence = self.kernel_operator(x, y, compute_divergence=True, kernel_hyperparams=kernel_hyperparams)
            return op.kernel_matrix(flatten), divergence
        op = self.kernel_operator(x, y, compute_divergence=False, kernel_hyperparams=kernel_hyperparams)
        return op.kernel_matrix(flatten)
