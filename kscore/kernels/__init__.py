# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
from kscore.utils import median_heuristic

from .adaptive_gaussian import CurlFreeAdaptiveGaussian, DiagonalAdaptiveGaussian
from .base import BaseKernel
from .curlfree_gaussian import CurlFreeGaussian
from .curlfree_imq import CurlFreeIMQ, CurlFreeIMQp
from .diagonal import Diagonal
from .diagonal_gaussian import DiagonalGaussian
from .diagonal_imq import DiagonalIMQ, DiagonalIMQp
from .square_curlfree import SquareCurlFree

__all__ = [
    "BaseKernel",
    "SquareCurlFree",
    "CurlFreeIMQ",
    "CurlFreeIMQp",
    "CurlFreeGaussian",
    "CurlFreeAdaptiveGaussian",
    "Diagonal",
    "DiagonalIMQ",
    "DiagonalIMQp",
    "DiagonalGaussian",
    "DiagonalAdaptiveGaussian",
    "median_heuristic",
]
