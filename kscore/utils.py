# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
from collections import namedtuple

import torch


def median_heuristic(x, y):
    # x: [..., n, d]
    # y: [..., m, d]
    # return: []
    n = x.shape[-2]
    m = y.shape[-2]
    x_expand = x.unsqueeze(-2)
    y_expand = y.unsqueeze(-3)
    pairwise_dist = torch.sqrt(torch.sum(torch.square(x_expand - y_expand), dim=-1))
    k = n * m // 2
    top_k_values = torch.topk(pairwise_dist.reshape(-1, n * m), k=k).values
    kernel_width = top_k_values[:, -1].reshape(x.shape[:-2]).detach()
    return float(kernel_width)


def random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    uniform_prob = torch.ones(inputs.shape[0])
    ind = torch.multinomial(uniform_prob, num_samples=n_samples, replacement=True)
    return inputs[ind]


def conjugate_gradient(operator, rhs, x=None, tol=1e-4, max_iter=40):
    """From tensorflow/contrib/solvers/linear_equations.py"""

    cg_state = namedtuple("CGState", ["i", "x", "r", "p", "gamma"])

    def stopping_criterion(i, state):
        return i < max_iter and torch.norm(state.r) > tol

    def cg_step(i, state):
        z = operator.apply(state.p)
        alpha = state.gamma / torch.sum(state.p * z)
        x = state.x + alpha * state.p
        r = state.r - alpha * z
        gamma = torch.sum(r * r)
        beta = gamma / state.gamma
        p = r + beta * state.p
        return i + 1, cg_state(i + 1, x, r, p, gamma)

    n = operator.shape[1:]
    rhs = rhs.unsqueeze(-1)
    if x is None:
        x = torch.zeros(n, dtype=rhs.dtype, device=rhs.device).unsqueeze(-1)
        r0 = rhs
    else:
        x = x.unsqueeze(-1)
        r0 = rhs - operator.apply(x)

    p0 = r0
    gamma0 = torch.sum(r0 * p0)
    tol *= torch.norm(r0)
    i = 0
    state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    while stopping_criterion(i, state):
        i, state = cg_step(i, state)

    return cg_state(state.i, x=state.x.squeeze(), r=state.r.squeeze(), p=state.p.squeeze(), gamma=state.gamma)
