# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class ThreeFcRegressionHead(nn.Module):
    """Conventional LayerNorm + 768-256-64-1 regression head."""

    def __init__(
            self, input_dim=768, hidden_dims=(256, 64), dropout=0.1):
        super().__init__()
        hidden_dim1, hidden_dim2 = (int(value) for value in hidden_dims)
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, 1)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        for layer in (self.fc1, self.fc2, self.output):
            trunc_normal_(layer.weight, std=2e-5)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        features = self.norm(features)
        features = self.dropout1(self.activation(self.fc1(features)))
        features = self.dropout2(self.activation(self.fc2(features)))
        return self.output(features)


class LinearRegressionHead(nn.Module):
    """LayerNorm + one linear SSC output layer."""

    def __init__(self, input_dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.output = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        trunc_normal_(self.output.weight, std=2e-5)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(self.norm(features))


class DeepFcRegressionHead(nn.Module):
    """LayerNorm followed by a configurable bottleneck MLP and SSC output."""

    def __init__(self, input_dim=768, hidden_dims=(256, 128, 64), dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        dimensions = (int(input_dim),) + tuple(int(value) for value in hidden_dims)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1))
        self.dropouts = nn.ModuleList(
            nn.Dropout(float(dropout)) for _ in self.hidden_layers)
        self.output = nn.Linear(dimensions[-1], 1)
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        for layer in (*self.hidden_layers, self.output):
            trunc_normal_(layer.weight, std=2e-5)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        features = self.norm(features)
        for layer, dropout in zip(self.hidden_layers, self.dropouts):
            features = dropout(self.activation(layer(features)))
        return self.output(features)


class ThreeFcVisionTransformer(VisionTransformer):
    """Ordinary ViT encoder with a conventional three-FC regressor."""

    def __init__(self, regression_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.head = ThreeFcRegressionHead(
            input_dim=int(kwargs["embed_dim"]),
            hidden_dims=(256, 64),
            dropout=regression_dropout)

    def reset_regression_head(self):
        self.head.reset_parameters()


class ConfigurableRegressionVisionTransformer(VisionTransformer):
    """Ordinary ViT encoder with a controlled regression-head variant."""

    def __init__(self, hidden_dims=None, regression_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        input_dim = int(kwargs["embed_dim"])
        if hidden_dims is None:
            self.head = LinearRegressionHead(input_dim=input_dim)
        else:
            self.head = DeepFcRegressionHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=regression_dropout)

    def reset_regression_head(self):
        self.head.reset_parameters()


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16_threefc_regression(**kwargs):
    model = ThreeFcVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        regression_dropout=0.1, **kwargs)
    return model


def vit_base_patch16_linear_regression(**kwargs):
    return ConfigurableRegressionVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hidden_dims=None, regression_dropout=0.1, **kwargs)


def vit_base_patch16_fourfc_regression(**kwargs):
    return ConfigurableRegressionVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hidden_dims=(256, 128, 64), regression_dropout=0.1, **kwargs)


def vit_base_patch16_fivefc_regression(**kwargs):
    return ConfigurableRegressionVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hidden_dims=(256, 128, 64, 32), regression_dropout=0.1, **kwargs)
