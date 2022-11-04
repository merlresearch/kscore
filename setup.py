# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    required = f.readlines()

setup(
    name="kscore",
    ext_modules=cythonize("kscore/kernels/utils.pyx"),
    include_dirs=[np.get_include()],
    version="1.0.0",
    install_requires=required,
    packages=find_packages(),
)
