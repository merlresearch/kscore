# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import torch

from .diagonal import Diagonal
from .square_curlfree import SquareCurlFree
from .utils import get_tsne_betas


def print_betas(betas):
    import math

    def sigma_from_beta(b):
        # b = 1 / (2 * s**2)
        # s**2 = 1 / (2 * b)
        # s = 1 / (sqrt(2 * b))
        return 1 / math.sqrt(2 * b)

    lo, mean, hi = betas.min().item(), betas.mean().item(), betas.max().item()
    s_lo, s_mean, s_hi = sigma_from_beta(lo), sigma_from_beta(mean), sigma_from_beta(hi)
    print(f"betas {lo:.3f}, {mean:.3f}, {hi:.3f}.\tsigmas {s_lo:.3f}, {s_mean:.3f}, {s_hi:.3f}")


class DiagonalAdaptiveGaussian(Diagonal):
    """NOTE - this kernel does not work well with the SpectralCutoff estimator"""

    def __init__(self):
        super().__init__(lambda *args: None)

    def __repr__(self):
        return f"{type(self).__name__}()"

    def _gram_impl(self, x, y, perplexity):
        r"""
        NOTE - since we use adaptive length scales, the naive K(x, y) != K(y, x).
        To be valid, our kernel matrix must be symmetric and positive semi-definite.

        Let alpha = ||x - y||^2

        We begin with:
            K(x, y) = exp(-alpha * beta_x)
            K(y, x) = exp(-alpha * beta_y)

        By averaging the log of the two kernels and then exponentiating again, we have:

            K_final(x, y) = K_final(y, x)  \
                & = exp[  (log(exp(-alpha * beta_x)) + log(exp(-alpha * beta_y)) / 2  ]
                & = exp[  ((-alpha * beta_x) + (-alpha * beta_y)) / 2  ]
                & = exp[  -alpha * (beta_x + beta_y) / 2  ]

        Notice this just corresponds to an RBF kernel with a beta = (beta_x + beta_y) / 2

        This function is still >= 0 by definition, and symmetric for (x, y) - so we have a valid kernel at the end
        """
        perplexity = perplexity or x.shape[0] - 10  # Default to perplexity = batch_size - 10

        n_train = x.shape[0]
        all_data = torch.cat((x, y))
        diff = all_data.unsqueeze(-2) - all_data.unsqueeze(-3)
        dist2 = torch.sum(diff * diff, dim=-1)
        betas_upper_right = torch.from_numpy(  # shape (n_train, n_test)
            get_tsne_betas(
                dist2.cpu().numpy()[:n_train, n_train:, ...],
                perplexity,
            )
        ).type(torch.float32)
        betas_lower_left = torch.from_numpy(  # shape (n_test, n_train)
            get_tsne_betas(
                dist2.cpu().numpy()[n_train:, :n_train, ...],
                perplexity,
            )
        ).type(torch.float32)
        betas_symmetric = (betas_upper_right + betas_lower_left.transpose(1, 0)) / 2
        betas_symmetric = betas_symmetric.to(x.device)
        rbf = torch.exp(-betas_symmetric * dist2[:n_train, n_train:, ...])  # [M, N]
        divergence = rbf.unsqueeze(-1) * (diff[:n_train, n_train:, ...] * 2 * betas_symmetric.unsqueeze(-1))
        # print_betas(betas_symmetric)
        return rbf, divergence


class CustomSquareCurlFree(SquareCurlFree):
    def _gram_derivatives(self, x, y, kernel_hyperparams):
        return self._gram_derivatives_impl(x, y, kernel_hyperparams)


class CurlFreeAdaptiveGaussian(CustomSquareCurlFree):
    def __init__(self):
        super().__init__(lambda *args: None)

    def __repr__(self):
        return f"{type(self).__name__}()"

    def _gram_derivatives_impl(self, x, y, perplexity):
        perplexity = perplexity or x.shape[0] - 10

        n_train = x.shape[0]
        all_data = torch.cat((x, y))
        diff = all_data.unsqueeze(-2) - all_data.unsqueeze(-3)
        dist2 = torch.sum(diff * diff, dim=-1)
        betas_upper_right = torch.from_numpy(  # shape (n_train, n_test)
            get_tsne_betas(
                dist2.cpu().numpy()[:n_train, n_train:, ...],
                perplexity,
            )
        ).type(torch.float32)
        betas_lower_left = torch.from_numpy(  # shape (n_test, n_train)
            get_tsne_betas(
                dist2.cpu().numpy()[n_train:, :n_train, ...],
                perplexity,
            )
        ).type(torch.float32)
        betas_symmetric = (betas_upper_right + betas_lower_left.transpose(1, 0)) / 2
        betas_symmetric = betas_symmetric.to(x.device)
        rbf = torch.exp(-betas_symmetric * dist2[:n_train, n_train:, ...])  # [M, N]
        G_1st = -rbf * betas_symmetric
        G_2nd = -G_1st * betas_symmetric
        G_3rd = -G_2nd * betas_symmetric
        # print_betas(betas_symmetric)
        return diff[:n_train, n_train:, ...], dist2[:n_train, n_train:, ...], G_1st, G_2nd, G_3rd
