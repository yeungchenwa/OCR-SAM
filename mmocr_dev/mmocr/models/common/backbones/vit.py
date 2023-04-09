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

import timm.models.vision_transformer
import torch
import torch.nn as nn
from mmocr.registry import MODELS


@MODELS.register_module()
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self,
                 global_pool=False,
                 patch_size=8,
                 img_size=(32, 128),
                 embed_dim=192,
                 depth=12,
                 num_heads=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained=None,
                 _2d_out=False,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            **kwargs)

        self.global_pool = global_pool
        self._2d_out = _2d_out
        self.patch_size = patch_size
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.reset_classifier(0)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % pretrained)
            checkpoint_model = checkpoint['model']
            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[
                        k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            msg = self.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # remove the cls token
        x = x[:, 1:]
        return x

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.forward_features(x)
        if self._2d_out:
            # N, L, C -> N, C, H, W
            x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim,
                                          H // self.patch_size,
                                          W // self.patch_size)
        return x


def vit_tiny_patch8(**kwargs):
    model = VisionTransformer(
        patch_size=8,
        img_size=(32, 128),
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=3.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_small_patch8(**kwargs):
    model = VisionTransformer(
        img_size=(32, 128),
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=(32, 128),
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
