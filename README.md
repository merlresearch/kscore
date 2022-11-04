<!--
Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)

SPDX-License-Identifier: BSD-3-Clause
SPDX-License-Identifier: MIT
-->
# Nonparametric Score Estimators (PyTorch reimplementation)

PyTorch reimplementation of code from "Nonparametric Score Estimators" (Yuhao Zhou, Jiaxin Shi, Jun Zhu. https://arxiv.org/abs/2005.10099).
See original Tensorflow implementation at https://github.com/miskcoo/kscore (MIT license). 

## Installation

Install project for local development with:
```shell
python3 -m venv venv
source venv/bin/activate

# Install from local source
pip install -Ue .
```

To install directly from GitHub:
```shell
source venv/bin/activate
pip install git+https://github.com/merlresearch/kscore.git
```

## Testing
Run tests with:
```shell
source venv/bin/activate
pytest tests/test_estimator.py -s
```

Run tests with coverage report:
(`term-missing` to show missing line ranges on terminal, `skip-covered` to ignore files with 100% coverage)
```shell
source venv/bin/activate
pytest --cov-report term-missing:skip-covered --cov=kscore tests/test_estimator.py
```

## Usage

Basic usage examples, as explained in the original Tensorflow implementation.

Create a score estimator:

```python
from kscore.estimators import Landweber, NuMethod, SpectralCutoff, Stein, Tikhonov
from kscore.kernels import (
    CurlFreeAdaptiveGaussian,
    CurlFreeGaussian,
    CurlFreeIMQ,
    CurlFreeIMQp,
    Diagonal,
    DiagonalAdaptiveGaussian,
    DiagonalGaussian,
    DiagonalIMQ,
    DiagonalIMQp,
    SquareCurlFree,
)

# Tikhonov regularization (Theorem 3.1), equivalent to KEF (Example 3.5)
kef_estimator = Tikhonov(lam=0.0001, use_cg=False, kernel=CurlFreeIMQ())

# Tikhonov regularization + Conjugate Gradient (KEF-CG, Example 3.8)
kefcg_estimator = Tikhonov(lam=0.0001, use_cg=True, kernel=CurlFreeIMQ())

# Tikhonov regularization + Nystrom approximation (Appendix C.1),
# equivalent to NKEF (Example C.1) using 60% samples
nkef_estimator = Tikhonov(lam=0.0001, use_cg=False, subsample_rate=0.6, kernel=CurlFreeIMQ())

# Tikhonov regularization + Nystrom approximation + Conjugate Gradient
nkefcg_estimator = Tikhonov(lam=0.0001, use_cg=True, subsample_rate=0.6, kernel=CurlFreeIMQ())

# Landweber iteration (Theorem 3.4)
landweber_estimator = Landweber(lam=0.00001, kernel=CurlFreeIMQ())
landweber_estimator = Landweber(iternum=100, kernel=CurlFreeIMQ())

# nu-method (Example C.4)
nu_estimator = NuMethod(lam=0.00001, kernel=CurlFreeIMQ())
nu_estimator = NuMethod(iternum=100, kernel=CurlFreeIMQ())

# Spectral cut-off regularization (Theorem 3.2),
# equivalent to SSGE (Example 3.6) using 90% eigenvalues
ssge_estimator = SpectralCutoff(keep_rate=0.9, kernel=DiagonalIMQ())

# Original Stein estimator
stein_estimator = Stein(lam=0.001)
```

Fit the score estimator using samples

```python
# manually specify the hyperparameter
estimator.fit(samples, kernel_hyperparams=kernel_width)
# automatically choose the hyperparameter (using the median trick)
estimator.fit(samples)
```

Predict the score

```python
gradient = estimator.compute_gradients(x)
```

Predict the energy (unnormalized log-density)

```python
log_p = estimator.compute_energy(x)   # only for curl-free kernels
```

Construct other curl-free kernels (see `kscore/kernels/curlfree_gaussian.py`)

## Effect of random seed

Note that the test suite is written with arbitrary thresholds on the L2 distance between estimated and true score function, and the cosine angle between estimated and true score function. 

The code in this repo is quite sensitive to random seed; this might be due to the particular set of test data constructed by `torch.randn`, or due to randomness within the estimators themselves.

The tests in the original Tensorflow repo are set with a particular random seed such that they all pass with that seed. 

By exploring a larger number of random seeds in the tests here, we see that the estimators meet the desired thresholds a majority of the time, but not always.

Finally - note that there is a difference between CPU and GPU randomness; simply changing the device where test data is placed will change the outcomes of the tests.

The effect of random seed and the differences between CPU and GPU behavior can be observed by changing the device for the test suite to GPU, and by increasing the number of random seeds from ~10 to >30

## Development

Install development requirements and pre-commit hooks.
Note that pre-commit hooks will be automatically run during `git commit`. If any
changes are made (such as automatically formatting a file), the commit step will be stopped
and the modified files should be inspected and staged (`git add -u` or equivalent) before
trying to commit again. 
```shell
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

To manually run pre-commit hooks at any time and inspect the resulting changes:
```shell
source venv/bin/activate
pre-commit run --all-files
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and License

Released under `BSD-3-Clause` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as listed below:

```
Copyright (c) 2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: BSD-3-Clause
```

`README.md` and files in `examples/`, `kscore/`, `tests/`, except `kscore/kernels/utils.pyx` were adapted 
from https://github.com/miskcoo/kscore (MIT license: see [LICENSES/MIT.txt](LICENSES/MIT.txt)):

```
Copyright (c) 2022 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)

SPDX-License-Identifier: BSD-3-Clause
SPDX-License-Identifier: MIT
```

`kscore/kernels/utils.pyx` was adapted from https://github.com/scikit-learn/scikit-learn/blob/0.24.2/sklearn/manifold/_utils.pyx 
(BSD-3-Clause license: see [LICENSES/BSD-3-Clause.md](LICENSES/BSD-3-Clause.md))

```
Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
Copyright (c) 2007-2021 The scikit-learn developers.

SPDX-License-Identifier: BSD-3-Clause
```
