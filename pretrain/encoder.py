# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for 1D signals (ECG, audio, time series) by HeartAge project.

import torch
import torch.nn as nn
from typing import List


_cur_active: torch.Tensor = None  # (B, 1, f) — active mask at smallest feature map resolution


def _get_active_ex_or_ii(L, returning_active_ex=True):
    """Expand the active mask to spatial size L, or return (batch_idx, pos_idx) of active positions."""
    repeat = L // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(repeat, dim=2)  # (B, 1, L)
    if returning_active_ex:
        return active_ex
    else:
        return active_ex.squeeze(1).nonzero(as_tuple=True)  # (bi, li)


# ---------------------------------------------------------------------------
# Sparse forward functions for 1D layers
# ---------------------------------------------------------------------------

def sp_conv_forward_1d(self, x: torch.Tensor):
    """Conv1d forward that zeros masked positions in the output."""
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(L=x.shape[2], returning_active_ex=True)  # (B,C,L) *= (B,1,L)
    return x


def sp_bn_forward_1d(self, x: torch.Tensor):
    """BatchNorm forward that only normalizes active (unmasked) positions."""
    ii = _get_active_ex_or_ii(L=x.shape[2], returning_active_ex=False)
    blc = x.permute(0, 2, 1)    # (B, C, L) -> (B, L, C)
    nc = blc[ii]                 # (N_active, C) — gather active features
    nc = super(type(self), self).forward(nc)  # BN1d on (N_active, C)
    out = torch.zeros_like(blc)
    out[ii] = nc                 # scatter back to (B, L, C)
    return out.permute(0, 2, 1)  # (B, C, L)


# ---------------------------------------------------------------------------
# Sparse layer classes (drop-in replacements with identical state_dict keys)
# ---------------------------------------------------------------------------

class SparseConv1d(nn.Conv1d):
    forward = sp_conv_forward_1d


class SparseMaxPooling1d(nn.MaxPool1d):
    forward = sp_conv_forward_1d


class SparseAvgPooling1d(nn.AvgPool1d):
    forward = sp_conv_forward_1d


class SparseBatchNorm1d(nn.BatchNorm1d):
    forward = sp_bn_forward_1d


class SparseSyncBatchNorm1d(nn.SyncBatchNorm):
    forward = sp_bn_forward_1d


# ---------------------------------------------------------------------------
# SparseEncoder1D: wraps any Conv1d-based model with sparse forward
# ---------------------------------------------------------------------------

class SparseEncoder1D(nn.Module):
    def __init__(self, cnn, input_size: int, sbn=False, verbose=False):
        super().__init__()
        self.downsample_ratio = cnn.get_downsample_ratio()
        self.enc_feat_map_chs = cnn.get_feature_map_channels()
        self.input_size = input_size
        self.sp_cnn = SparseEncoder1D.dense_model_to_sparse(
            m=cnn, verbose=verbose, sbn=sbn,
        )

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        """Recursively replace Conv1d/BN1d/Pool1d with sparse versions."""
        oup = m
        if isinstance(m, nn.Conv1d):
            bias = m.bias is not None
            oup = SparseConv1d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias,
                padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool1d):
            oup = SparseMaxPooling1d(
                m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, return_indices=m.return_indices,
                ceil_mode=m.ceil_mode,
            )
        elif isinstance(m, nn.AvgPool1d):
            oup = SparseAvgPooling1d(
                m.kernel_size, m.stride, m.padding,
                ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad,
            )
        elif isinstance(m, nn.SyncBatchNorm):
            oup = SparseSyncBatchNorm1d(
                m.weight.shape[0], eps=m.eps, momentum=m.momentum,
                affine=m.affine, track_running_stats=m.track_running_stats,
            )
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        elif isinstance(m, nn.BatchNorm1d):
            oup = (SparseSyncBatchNorm1d if sbn else SparseBatchNorm1d)(
                m.weight.shape[0], eps=m.eps, momentum=m.momentum,
                affine=m.affine, track_running_stats=m.track_running_stats,
            )
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        elif isinstance(m, nn.Conv2d):
            raise NotImplementedError(
                f"SparK1D does not support 2D layers: {type(m)}"
            )

        for name, child in m.named_children():
            oup.add_module(
                name,
                SparseEncoder1D.dense_model_to_sparse(child, verbose=verbose, sbn=sbn),
            )
        del m
        return oup

    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)
