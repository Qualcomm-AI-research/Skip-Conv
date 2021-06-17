# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

import torch


def roll_time(x: torch.Tensor) -> torch.Tensor:
    return x.view((-1,) + x.shape[2:])


def unroll_time(x: torch.Tensor, t: int) -> torch.Tensor:
    return x.view(
        (
            -1,
            t,
        )
        + x.shape[1:]
    )
