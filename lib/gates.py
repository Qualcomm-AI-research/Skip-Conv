# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from lib.gumbel_module import GumbelSigmoid


class GateType(Enum):
    NORM_GATE = "norm-gate"
    GUMBEL_GATE = "gumbel-gate"


class NormGateType(Enum):
    INPUT = "input"
    OUTPUT = "output"


class NormGate(nn.Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
        type: NormGateType = NormGateType.OUTPUT,
        eps: float = 1e-1,
        norm: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.type = type
        self.norm = float(norm)
        self.eps = eps
        self.qbin = 1e5

    def forward(self, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_train(r, w)
        else:
            return self._forward_test(r, w)

    def _forward_train(self, r, w):
        n, c, h, w = r.shape
        return torch.ones(n, 1, h, w).to(r.device)  # no gating during training

    def _forward_test(self, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.type is NormGateType.INPUT:
            return self._forward_input(r)
        elif self.type is NormGateType.OUTPUT:
            return self._forward_output(r, w)
        raise ValueError

    def _forward_input(self, r: torch.Tensor) -> torch.Tensor:
        """Input norm gates, Eq (5)"""
        ri = F.avg_pool2d(r.abs(), self.kernel_size, self.stride, self.padding)
        ri_norm = torch.norm(ri, p=self.norm, dim=1, keepdim=True) / ri.size(2)
        ri_out_discrete = (ri_norm * self.qbin).floor() / self.qbin

        return (torch.sign(ri_out_discrete - self.eps) + 1) / 2

    def _forward_output(self, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Output norm gates, Eq (7)"""
        ri = F.avg_pool2d(r.abs(), self.kernel_size, self.stride, self.padding)
        ri_norm = torch.norm(ri, p=self.norm, dim=1, keepdim=True) / ri.size(2)
        w_norm = torch.norm(w, p=self.norm) / w.numel()
        ri_out = ri_norm * w_norm
        ri_out_discrete = (ri_out * self.qbin).floor() / self.qbin

        return (torch.sign(ri_out_discrete - self.eps) + 1) / 2

    def get_mac(self, r: torch.Tensor, g: torch.Tensor) -> int:
        return 0


class GumbelGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        structure: int = 1,
    ):
        super(GumbelGate, self).__init__()
        self.gs = GumbelSigmoid()
        self.structure = structure
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        gating_layers = [
            nn.Conv2d(
                in_channels, 1, kernel_size=kernel_size, stride=stride, padding=padding
            ),
        ]

        assert self.structure in [1, 2, 4, 8]
        if self.structure == 1:
            structure_layers = []
        else:
            structure_layers = [
                nn.MaxPool2d(kernel_size=self.structure, stride=self.structure),
                nn.UpsamplingNearest2d(scale_factor=self.structure),
            ]

        self.gate_network = nn.Sequential(*gating_layers, *structure_layers)

        self.init_weights()

    def init_weights(self, gate_bias_init: float = 0.6) -> None:
        conv = self.gate_network[0]
        torch.nn.init.xavier_uniform_(conv.weight)
        conv.bias.data.fill_(gate_bias_init)

    def forward(self, gate_inp: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """Gumbel gates, Eq (8)"""
        pi_log = self.gate_network(gate_inp)
        return self.gs(pi_log, force_hard=True)

    def get_mac(self, r: torch.Tensor, g: torch.Tensor) -> int:
        n, c_in, h_in, w_in = r.shape
        n, _, h_out, w_out = g.shape
        if isinstance(self.kernel_size, tuple):
            k_h, k_w = self.kernel_size
        else:
            k_h = k_w = self.kernel_size

        mac_gates = n * h_out * w_out * c_in * 1 * k_h * k_w

        return mac_gates
