# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for 1D signals (ECG, audio, time series) by HeartAge project.

import math
from typing import List

import torch
import torch.nn as nn


def _is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


class UNetBlock1D(nn.Module):
    """UNet block with 2x upsampling for 1D signals."""

    def __init__(self, cin, cout, bn_layer):
        super().__init__()
        self.up_sample = nn.ConvTranspose1d(
            cin, cin, kernel_size=4, stride=2, padding=1, bias=True,
        )
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            bn_layer(cin),
            nn.ReLU6(inplace=True),
            nn.Conv1d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            bn_layer(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder1D(nn.Module):
    """Hierarchical UNet decoder for 1D signals.

    Takes multi-scale features from the encoder (via densify projections)
    and progressively upsamples to reconstruct the original signal.

    Args:
        up_sample_ratio: total spatial downsample ratio of the encoder (must be power of 2)
        width: channel width of the first (deepest) decoder block
        out_channels: number of output channels (e.g. 12 for 12-lead ECG)
        sbn: use SyncBatchNorm (for multi-GPU)
    """

    def __init__(self, up_sample_ratio, width=512, out_channels=12, sbn=False):
        super().__init__()
        self.width = width
        assert _is_pow2n(up_sample_ratio), \
            f"up_sample_ratio must be power of 2, got {up_sample_ratio}"
        n = round(math.log2(up_sample_ratio))
        # Channel halving rule: [width, width//2, width//4, ..., width//2^n]
        channels = [self.width // 2 ** i for i in range(n + 1)]
        bn_layer = nn.SyncBatchNorm if sbn else nn.BatchNorm1d
        self.dec = nn.ModuleList([
            UNetBlock1D(cin, cout, bn_layer)
            for cin, cout in zip(channels[:-1], channels[1:])
        ])
        self.proj = nn.Conv1d(channels[-1], out_channels, kernel_size=1, bias=True)
        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
