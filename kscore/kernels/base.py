#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class BaseKernel:

    def __init__(self, kernel_type, kernel_hyperparams, heuristic_hyperparams):
        if kernel_hyperparams is not None:
            heuristic_hyperparams = lambda x, y: kernel_hyperparams
        self._kernel_type = kernel_type
        self._heuristic_hyperparams = heuristic_hyperparams

    def kernel_type(self):
        return self._kernel_type

    def heuristic_hyperparams(self, x, y):
        return self._heuristic_hyperparams(x, y)

    def kernel_operator(self, x, y, kernel_hyperparams, **kwargs):
        pass

    def kernel_matrix(self, x, y, kernel_hyperparams, **kwargs):
        pass
