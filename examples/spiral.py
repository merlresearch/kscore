# Copyright (c) 2022 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 Yuhao Zhou (yuhaoz.cs@gmail.com), Jiaxin Shi (ishijiaxin@126.com)
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-License-Identifier: MIT
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import (
    add_estimator_params,
    clip_energy,
    get_estimator,
    linspace_2d,
    plot_vector_field,
)


def generate_data(n_samples):
    theta = torch.FloatTensor(n_samples).uniform_(3.0, 15.0)
    noise = torch.randn(n_samples, 2) * np.exp(-1.0)
    samples = torch.stack((-2.0 + 2 * theta * torch.cos(theta), 2 * theta * torch.sin(theta)), 1) + noise
    return samples


def main(args):
    torch.random.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    kernel_width = 8.0
    n_samples = args.n_samples
    size, energy_size = 25, 300
    lower_box, upper_box = -args.plot_range, args.plot_range

    samples = generate_data(n_samples)
    x = linspace_2d(size, lower_box, upper_box)
    x_energy = linspace_2d(energy_size, lower_box, upper_box)

    estimator = get_estimator(args)
    estimator.fit(samples, kernel_hyperparams=kernel_width)

    gradient = estimator.compute_gradients(x)
    if "curlfree" in args.kernel:
        energy = estimator.compute_energy(x_energy)
    else:
        energy = 0.0

    # plot energy
    if "curlfree" in args.kernel:
        plt.figure(figsize=(4, 4))
        if args.clip_energy:
            energy = clip_energy(energy, threshold=args.clip_threshold)
        energy = energy.numpy()
        img = np.transpose(np.reshape(energy, [energy_size, energy_size]))
        img = np.flip(img, axis=0)
        plt.imshow(img, extent=[lower_box, upper_box, lower_box, upper_box])
        plt.savefig("spiral-density.png")
        plt.close()

    # plot the score field
    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:, 0], samples[:, 1], 2)
    plot_vector_field(x, gradient)
    plt.savefig("spiral-gradient.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_estimator_params(parser)
    parser.add_argument("--n_samples", type=int, default=200, help="sample size.")
    parser.add_argument("--plot_range", default=32, type=int)
    parser.add_argument("--clip_energy", default=True, type=bool, help="whether to clip the energy function.")
    parser.add_argument("--clip_threshold", default=24, type=int)
    args = parser.parse_args(sys.argv[1:])

    main(args)
