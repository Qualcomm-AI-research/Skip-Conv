# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

import torch

from lib.gates import GateType
from lib.gates import NormGateType
from lib.skip_conv import SkipConv2d

device = "cuda" if torch.cuda.is_available() else "cpu"

b, t, c, h, w = 1, 8, 32, 224, 224

conv_ops = {
    "gate_type": GateType.GUMBEL_GATE,
    "in_channels": c,
    "out_channels": 64,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "norm_gate_type": NormGateType.OUTPUT,
    "norm_gate_eps": 1e-1,
    "gumbel_gate_structure": 2,
}


def forward_train(model: SkipConv2d, x: torch.Tensor) -> None:
    """
    During training, the Skip-Convolution is fed with clips of t frames.
    As such, the input tensor has shape (batchsize, n_frames, channels, height, width).
    The model is stateless in training mode.

    :param model: the skip-convolution module.
    :param x: input tensor having shape (batchsize, n_frames, channels, height, width).
    """
    model = model.train()

    y = model(x)
    print(y.shape)


def forward_test(model: SkipConv2d, x: torch.Tensor, reset_every: int = 4) -> None:
    """
    During test, a sequence of t frames is fed iteratively in a for loop.
    As such, the input tensor has shape (batchsize, channels, height, width).
    The model is stateful in eval mode, and it stores the previous input and output tensors.
    Every `reset_every` frames, the state is reset, and a new reference frame is instantiated.

    :param model: the skip-convolution module.
    :param x: input tensor having shape (batchsize, n_frames, channels, height, width).
    :param reset_every: interval between reference frames.
    """
    model = model.eval()

    y = []
    for frame_idx in range(x.shape[1]):
        if frame_idx % reset_every == 0:
            model.reset()

        y.append(model(x[:, frame_idx]))

    y = torch.stack(y, dim=1)
    print(y.shape)


def main():
    """
    Main function.
    The script will call two functions showcasing how the operator should be used
    in training (stateless, cumsum operation) and testing (stateful) within a backbone network.
    The reported example feeds random tensors and prints out the resulting shapes.
    """
    model = SkipConv2d(**conv_ops).to(device)
    x = torch.rand(b, t, c, h, w).to(device)

    forward_train(model, x)
    forward_test(model, x)


if __name__ == "__main__":
    main()
