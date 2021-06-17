# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from lib import gates
from lib.gates import GateType
from lib.gates import NormGateType
from lib.utils import roll_time
from lib.utils import unroll_time


class SkipConv2d(nn.Conv2d):
    def __init__(
        self,
        gate_type: GateType,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_gate_type: NormGateType = NormGateType.OUTPUT,
        norm_gate_eps: float = 1e-1,
        gumbel_gate_structure: int = 1,
    ):
        super(SkipConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self.gate_type = gate_type
        if self.gate_type is GateType.NORM_GATE:
            self.norm_gate_type = norm_gate_type
            self.norm_gate_eps = norm_gate_eps
            self.gate = gates.NormGate(
                kernel_size, stride, padding, self.norm_gate_type, self.norm_gate_eps
            )
        elif self.gate_type is GateType.GUMBEL_GATE:
            self.gumbel_gate_structure = gumbel_gate_structure
            self.gate = gates.GumbelGate(
                in_channels, kernel_size, stride, padding, self.gumbel_gate_structure
            )
        else:
            raise ValueError

        self.z0 = None
        self.x0 = None
        self.mac = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        t = x.shape[1]

        x0 = x[:, 0]
        z0 = super(SkipConv2d, self).forward(x0)

        r = roll_time(x[:, 1:] - x[:, :-1])
        zr = super(SkipConv2d, self).forward(r)
        g = self.gate(r.abs(), self.weight)
        zr = zr * g

        z0 = unroll_time(z0, t=1)
        zr = unroll_time(zr, t=t - 1)
        z = torch.cat((z0, zr), dim=1)
        z = z.cumsum(dim=1)

        self.mac = self._get_mac_train(r, z, g)
        return z

    def _forward_test(self, x: torch.Tensor) -> torch.Tensor:
        if self.x0 is None:
            z = super(SkipConv2d, self).forward(x)
            mac = self._get_mac_test_reference(z)
        else:
            x0, z0 = self.x0, self.z0
            r = x - x0
            g = self.gate(r.abs(), self.weight)
            zr = super(SkipConv2d, self).forward(r)
            zr = zr * g
            z = z0 + zr
            mac = self._get_mac_test_residual(r, z, g)

        self.x0 = x
        self.z0 = z
        self.mac = mac
        return z

    def reset(self) -> None:
        assert (
            not self.training
        ), "reset() method should not be called in training mode."
        # Resets state, used for test.
        self.z0 = None
        self.x0 = None
        self.mac = None

    def eval(self):
        # Sets the model in evaluation mode, and also resets the state.
        ret = super(SkipConv2d, self).eval()
        self.reset()
        return ret

    # -----------------------
    # MAC computing functions
    # -----------------------

    def _get_mac_train(self, r: torch.Tensor, z: torch.Tensor, g: torch.Tensor) -> int:
        n, t, c_out, h_out, w_out = z.shape
        _, c_in, k_h, k_w = self.weight.shape

        mac_ref = 1 * h_out * w_out * c_in * c_out * k_h * k_w
        mac_res = g.sum().item() * c_in * c_out * k_h * k_w
        mac_gat = self.gate.get_mac(r, g)
        return mac_ref + mac_res + mac_gat

    def _get_mac_test_reference(self, z: torch.Tensor) -> int:
        n, c_out, h_out, w_out = z.shape
        _, c_in, k_h, k_w = self.weight.shape

        assert n == 1
        mac_ref = n * h_out * w_out * c_in * c_out * k_h * k_w
        return mac_ref

    def _get_mac_test_residual(
        self, r: torch.Tensor, z: torch.Tensor, g: torch.Tensor
    ) -> int:
        _, C_out, Hout, Wout = z.shape
        _, C_in, Kh, Kw = self.weight.shape

        mac_res = g.sum().item() * C_in * C_out * Kh * Kw
        mac_gat = self.gate.get_mac(r, g)
        return mac_res + mac_gat
