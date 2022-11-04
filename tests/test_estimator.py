# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
from functools import partial

import pytest
import torch

from kscore.estimators import Landweber, NuMethod, SpectralCutoff, Stein, Tikhonov
from kscore.kernels import (
    CurlFreeAdaptiveGaussian,
    CurlFreeGaussian,
    CurlFreeIMQ,
    DiagonalAdaptiveGaussian,
    DiagonalGaussian,
    DiagonalIMQ,
    median_heuristic,
)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def heuristic():
    def _heuristic(x, y):
        return 1.5 * median_heuristic(x, y)

    return _heuristic


@pytest.fixture
def diagonal_kernel(heuristic):
    return DiagonalIMQ(heuristic_hyperparams=heuristic)


@pytest.fixture
def curlfree_kernel(heuristic):
    return CurlFreeIMQ(heuristic_hyperparams=heuristic)


def _test(estimator, l2_bound, cos_bound, kernel_hyperparams=None, n_seeds=10, device=DEFAULT_DEVICE, threshold=0.8):
    """at least `threshold`*`n_seeds` tries must pass the specified bounds"""
    if kernel_hyperparams is None:
        kernel_hyperparams = [None, 3.0]

    def _test0(kernel_hyperparams, seed):
        torch.manual_seed(seed)

        x_train = torch.randn(256, 2, device=device)
        x_test = torch.randn(256, 2, device=device)
        y_ans = -0.5 * x_test

        estimator.fit(x_train, kernel_hyperparams=kernel_hyperparams)
        y_test = estimator.compute_gradients(x_test)

        l2_dist = torch.norm(y_test - y_ans, dim=-1)
        l2_dist /= torch.norm(y_ans, dim=-1) + 1
        l2_dist = torch.mean(l2_dist)

        cos_dist = torch.sum(y_test * y_ans, dim=-1)
        cos_dist /= torch.norm(y_ans, dim=-1) * torch.norm(y_test, dim=-1)
        cos_dist = torch.mean(cos_dist)

        return l2_dist, cos_dist

    total = 0
    l2_success_total = 0
    cos_success_total = 0
    l2_dists = []
    cos_dists = []
    for seed in range(n_seeds):
        for kernel_hyperparam in kernel_hyperparams:
            l2_dist, cos_dist = _test0(kernel_hyperparam, seed)

            l2_dists.append(format(l2_dist, ".3f"))
            cos_dists.append(format(cos_dist, ".3f"))

            l2_success_total += 1 if l2_dist < l2_bound else 0
            cos_success_total += 1 if cos_dist > cos_bound else 0
            total += 1

    pass_thresh = max(1, int(threshold * total))

    msg1 = "\n".join(
        [
            f"{estimator}",
            f"L2 success: {l2_success_total}/{total}",
            f"pass_thresh: {pass_thresh}",
            f"upper bound: {l2_bound}",
            f"dists: {l2_dists}",
        ]
    )
    msg2 = "\n".join(
        [
            f"{estimator}",
            f"Cos success: {cos_success_total}/{total}",
            f"pass_thresh: {pass_thresh}",
            f"lower bound: {cos_bound}",
            f"dists: {cos_dists}",
        ]
    )
    assert l2_success_total >= pass_thresh, msg1
    assert cos_success_total >= pass_thresh, msg2


def test_smoketest_all_pairs_cpu(heuristic):
    """Run each pair with very permissive thresholds, default parameters, and median heuristic hyperparams"""
    for estimator in [
        partial(Landweber, lam=0.01),
        partial(NuMethod, lam=0.1),
        partial(SpectralCutoff, lam=0.005),
        partial(Stein, lam=0.001),
        partial(Tikhonov, lam=0.1),
    ]:
        for kernel in [
            CurlFreeGaussian(heuristic_hyperparams=heuristic),
            CurlFreeIMQ(heuristic_hyperparams=heuristic),
            CurlFreeAdaptiveGaussian(),
            DiagonalGaussian(heuristic_hyperparams=heuristic),
            DiagonalIMQ(heuristic_hyperparams=heuristic),
            DiagonalAdaptiveGaussian(),
        ]:
            try:
                _test(estimator(kernel=kernel), l2_bound=0.8, cos_bound=0.2, kernel_hyperparams=[None], n_seeds=1)
            except NotImplementedError:
                continue


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_smoketest_all_pairs_gpu(heuristic):
    """Run each pair on GPU with very permissive thresholds, default parameters, and median heuristic hyperparams"""
    for estimator in [
        partial(Landweber, lam=0.01),
        partial(NuMethod, lam=0.1),
        partial(SpectralCutoff, lam=0.005),
        partial(Stein, lam=0.001),
        partial(Tikhonov, lam=0.1),
    ]:
        for kernel in [
            CurlFreeGaussian(heuristic_hyperparams=heuristic),
            CurlFreeIMQ(heuristic_hyperparams=heuristic),
            CurlFreeAdaptiveGaussian(),
            DiagonalGaussian(heuristic_hyperparams=heuristic),
            DiagonalIMQ(heuristic_hyperparams=heuristic),
            DiagonalAdaptiveGaussian(),
        ]:
            try:
                _test(
                    estimator(kernel=kernel),
                    l2_bound=0.8,
                    cos_bound=0.2,
                    kernel_hyperparams=[None],
                    n_seeds=1,
                    device=torch.device("cuda"),
                )
            except NotImplementedError:
                continue


def _test_tikhonov(subsample_rate, truncated_tikhonov, curlfree_kernel, diagonal_kernel):
    est_curlfree_cg = Tikhonov(
        lam=0.1,
        kernel=curlfree_kernel,
        truncated_tikhonov=truncated_tikhonov,
        subsample_rate=subsample_rate,
        use_cg=True,
        maxiter_cg=100,
    )

    est_curlfree_nocg = Tikhonov(
        lam=0.1,
        kernel=curlfree_kernel,
        truncated_tikhonov=truncated_tikhonov,
        subsample_rate=subsample_rate,
        use_cg=False,
    )

    est_diagonal_cg = Tikhonov(
        lam=0.1,
        kernel=diagonal_kernel,
        truncated_tikhonov=truncated_tikhonov,
        subsample_rate=subsample_rate,
        use_cg=True,
        maxiter_cg=100,
    )

    est_diagonal_nocg = Tikhonov(
        lam=0.1,
        kernel=diagonal_kernel,
        truncated_tikhonov=truncated_tikhonov,
        subsample_rate=subsample_rate,
        use_cg=False,
    )

    _test(est_curlfree_cg, 0.25, 0.95)
    _test(est_diagonal_cg, 0.25, 0.95)
    _test(est_curlfree_nocg, 0.25, 0.95)
    _test(est_diagonal_nocg, 0.25, 0.95)


def test_tikhonov(curlfree_kernel, diagonal_kernel):
    _test_tikhonov(None, False, curlfree_kernel, diagonal_kernel)


def test_tikhonov_nystrom(curlfree_kernel, diagonal_kernel):
    _test_tikhonov(0.7, False, curlfree_kernel, diagonal_kernel)
    _test_tikhonov(0.7, True, curlfree_kernel, diagonal_kernel)


def test_landweber(curlfree_kernel, diagonal_kernel):
    _test(Landweber(lam=0.001, iternum=None, kernel=curlfree_kernel), 0.5, 0.95)
    _test(Landweber(lam=0.1, iternum=None, kernel=diagonal_kernel), 0.5, 0.95)
    _test(Landweber(lam=None, iternum=1000, kernel=curlfree_kernel), 0.5, 0.95)
    _test(Landweber(lam=None, iternum=10, kernel=diagonal_kernel), 0.5, 0.95)


def test_nu_method(curlfree_kernel, diagonal_kernel):
    def _test_nu(nu, lam, iternum):
        _test(NuMethod(lam=lam, iternum=iternum, nu=nu, kernel=curlfree_kernel), 0.25, 0.95)
        _test(NuMethod(lam=lam, iternum=iternum, nu=nu, kernel=diagonal_kernel), 0.25, 0.95)

    _test_nu(1.0, None, 5)
    _test_nu(1.0, 0.1, None)
    _test_nu(2.0, None, 5)
    _test_nu(2.0, 0.1, None)


def test_spectral_cutoff(curlfree_kernel, diagonal_kernel):
    # This is sensitive to parameters
    _test(SpectralCutoff(lam=0.005, keep_rate=None, kernel=curlfree_kernel), 0.6, 0.85)
    _test(SpectralCutoff(lam=0.005, keep_rate=None, kernel=diagonal_kernel), 0.6, 0.85)


def test_stein(diagonal_kernel):
    _test(Stein(lam=0.001, kernel=diagonal_kernel), 0.3, 0.95)


def test_gaussian_curlfree_nu():
    def _test_local(nu, lam, iternum):
        _test(NuMethod(lam=lam, iternum=iternum, nu=nu, kernel=CurlFreeGaussian()), 0.5, 0.95)

    _test_local(1.0, None, 5)
    _test_local(1.0, 0.1, None)
    _test_local(2.0, None, 5)
    _test_local(2.0, 0.1, None)
    _test_local(10.0, 0.01, None)


def test_gaussian_diagonal():
    kernel = DiagonalGaussian
    for est in [
        Stein(lam=1e-5, kernel=kernel()),
        NuMethod(nu=1.0, lam=None, iternum=5, kernel=kernel()),
        Landweber(lam=1e-2, kernel=kernel()),
        SpectralCutoff(lam=5e-3, kernel=kernel()),
        Tikhonov(lam=0.1, kernel=kernel()),
    ]:
        _test(est, 0.5, 0.9, kernel_hyperparams=[None], n_seeds=10)


def test_adaptive_gaussian_diagonal():
    kernel = DiagonalAdaptiveGaussian
    for est in [
        Stein(lam=1e-3, kernel=kernel()),
        NuMethod(lam=0.1, kernel=kernel()),
        Landweber(lam=1e-2, kernel=kernel()),
        Tikhonov(lam=0.1, kernel=kernel()),
    ]:
        _test(est, 0.4, 0.95, kernel_hyperparams=[None], n_seeds=10)


def test_adaptive_gaussian_curlfree():
    kernel = CurlFreeAdaptiveGaussian
    for est in [
        NuMethod(lam=0.1, kernel=kernel()),
        Landweber(lam=1e-2, kernel=kernel()),
        Tikhonov(lam=0.1, kernel=kernel()),
    ]:
        _test(est, 0.4, 0.95, kernel_hyperparams=[None], n_seeds=10)
