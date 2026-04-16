# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted for 1D signals (ECG, audio, time series) by HeartAge project.

from pprint import pformat
from typing import List
import sys

import torch
import torch.nn as nn

from . import encoder as encoder_module
from .decoder import LightDecoder1D


class SparK1D(nn.Module):
    """Sparse and Hierarchical Masked Modeling for 1D ConvNets.

    Workflow:
      1. Mask random patches at feature-map resolution
      2. Encode with sparse convolutions (masked positions zeroed after every layer)
      3. Densify: fill masked positions with learnable mask tokens
      4. Decode with hierarchical UNet (multi-scale skip connections)
      5. Loss: per-patch-normalized MSE on masked patches only
    """

    def __init__(
        self,
        sparse_encoder: encoder_module.SparseEncoder1D,
        dense_decoder: LightDecoder1D,
        mask_ratio=0.6,
        input_size=4992,
        densify_norm='bn',
        sbn=False,
    ):
        super().__init__()
        downsample_ratio = sparse_encoder.downsample_ratio
        self.downsample_ratio = downsample_ratio
        self.fmap_len = input_size // downsample_ratio
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_len * (1 - mask_ratio))

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder

        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        # Build densify layers (from smallest to largest feature map)
        e_widths = list(sparse_encoder.enc_feat_map_chs)  # copy
        d_width = self.dense_decoder.width
        for i in range(self.hierarchy):
            e_width = e_widths.pop()

            # Learnable mask token for this scale
            p = nn.Parameter(torch.zeros(1, e_width, 1))
            nn.init.trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)

            # Densify normalization
            if self.densify_norm_str == 'bn':
                dnorm = (encoder_module.SparseSyncBatchNorm1d if self.sbn
                         else encoder_module.SparseBatchNorm1d)(e_width)
            else:
                dnorm = nn.Identity()
            self.densify_norms.append(dnorm)

            # Densify projection (map encoder channels → decoder channels)
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()
            else:
                ks = 1 if i == 0 else 3
                densify_proj = nn.Conv1d(
                    e_width, d_width, kernel_size=ks,
                    stride=1, padding=ks // 2, bias=True,
                )
            self.densify_projs.append(densify_proj)

            d_width //= 2

    def mask(self, B: int, device, generator=None):
        """Generate random patch-level mask. Returns (B, 1, fmap_len) bool tensor."""
        f = self.fmap_len
        idx = torch.rand(B, f, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)
        return torch.zeros(B, f, dtype=torch.bool, device=device).scatter_(
            dim=1, index=idx, value=True,
        ).view(B, 1, f)

    def forward(self, inp_bcl: torch.Tensor, active_b1f=None):
        """
        Args:
            inp_bcl: (B, C, L) input signal (e.g. 12-lead ECG)
            active_b1f: optional pre-computed mask (B, 1, fmap_len), True=keep

        Returns:
            recon_loss: scalar, per-patch-normalized MSE on masked positions
        """
        # Step 1: Mask
        if active_b1f is None:
            active_b1f = self.mask(inp_bcl.shape[0], inp_bcl.device)
        encoder_module._cur_active = active_b1f  # (B, 1, f)
        # Expand mask to input resolution
        active_b1l = active_b1f.repeat_interleave(
            self.downsample_ratio, dim=2,
        )  # (B, 1, L)
        masked_bcl = inp_bcl * active_b1l

        # Step 2: Sparse encode → hierarchical features
        fea_list: List[torch.Tensor] = self.sparse_encoder(masked_bcl)
        fea_list.reverse()  # now: smallest → largest feature map

        # Step 3: Densify — fill masked positions with mask tokens
        cur_active = active_b1f  # (B, 1, f)
        to_dec = []
        for i, bcf in enumerate(fea_list):
            if bcf is not None:
                bcf = self.densify_norms[i](bcf)
                mask_tokens = self.mask_tokens[i].expand_as(bcf)
                bcf = torch.where(
                    cur_active.expand_as(bcf), bcf, mask_tokens,
                )
                bcf = self.densify_projs[i](bcf)
            to_dec.append(bcf)
            cur_active = cur_active.repeat_interleave(2, dim=2)

        # Step 4: Decode
        rec_bcl = self.dense_decoder(to_dec)

        # Step 5: Per-patch-normalized MSE loss on masked positions
        inp = self._patchify(inp_bcl)  # (B, n_patches, C * patch_size)
        rec = self._patchify(rec_bcl)

        # Compute loss in FP32 for numerical stability
        inp = inp.float()
        rec = rec.float()
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var

        l2_loss = ((rec - inp) ** 2).mean(dim=2)  # (B, n_patches)
        non_active = active_b1f.logical_not().int().view(
            active_b1f.shape[0], -1,
        )  # (B, n_patches)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)

        return recon_loss

    def _patchify(self, bcl):
        """(B, C, L) → (B, n_patches, C * patch_size)"""
        p = self.downsample_ratio
        f = self.fmap_len
        B, C = bcl.shape[:2]
        bcl = bcl.reshape(B, C, f, p)           # (B, C, f, p)
        bcl = bcl.permute(0, 2, 3, 1)           # (B, f, p, C)
        return bcl.reshape(B, f, C * p)          # (B, f, C*p)

    def _unpatchify(self, bfn):
        """(B, n_patches, C * patch_size) → (B, C, L)"""
        p = self.downsample_ratio
        f = self.fmap_len
        B = bfn.shape[0]
        C = bfn.shape[-1] // p
        bfn = bfn.reshape(B, f, p, C)           # (B, f, p, C)
        bfn = bfn.permute(0, 3, 1, 2)           # (B, C, f, p)
        return bfn.reshape(B, C, f * p)          # (B, C, L)

    def get_encoder_state_dict(self):
        """Extract encoder weights for downstream fine-tuning.

        Returns state_dict with same keys as the original (dense) CNN.
        """
        return self.sparse_encoder.sp_cnn.state_dict()

    def __repr__(self):
        return (
            f'\n[SparK1D.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK1D.structure]: '
            f'{super(SparK1D, self).__repr__().replace(SparK1D.__name__, "")}'
        )

    def get_config(self):
        return {
            'input_size': self.input_size,
            'mask_ratio': self.mask_ratio,
            'fmap_len': self.fmap_len,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn,
            'hierarchy': self.hierarchy,
            'downsample_ratio': self.downsample_ratio,
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            'dense_decoder.width': self.dense_decoder.width,
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False,
                   with_config=False):
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars,
        )
        if with_config:
            state['config'] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config = state_dict.pop('config', None)
        incompatible = super().load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = (f'[SparK1D.load_state_dict] config mismatch: '
                           f'this.{k}={v} (ckpt.{k}={ckpt_v})')
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible
