# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from operator import is_
import torch
import torch.nn as nn
from functools import partial

from .deit_vision_transformer import VisionTransformer, _cfg

# from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224'
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attn_matrixs = []
        intermediate_features = []
        for block in self.blocks:
            x, attn_matrix = block(x)
            attn_matrixs.append(attn_matrix)
            intermediate_features.append(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), attn_matrixs, intermediate_features
        else:
            return x[:, 0], x[:, 1], attn_matrixs, intermediate_features


    def forward(self, x):
        x = self.forward_features(x)
        
        if self.head_dist is not None:
            cls_x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                return (cls_x, x_dist), x[2]
            else:
                return (cls_x + x_dist) / 2, x[2]
        else:
            cls_x = self.head(x[0])
            return cls_x, x[1] #




@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer= nn.GELU, **kwargs)
    model.default_cfg = _cfg()
    if pretrained and kwargs['num_classes'] == 1000:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        print("load pretrained")
        model.load_state_dict(checkpoint["model"])
    elif pretrained and kwargs['num_classes'] == 100:
        raise ValueError('No trained model provided')
    return model

@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer= nn.GELU, **kwargs)
    model.default_cfg = _cfg()
    if pretrained and kwargs['num_classes'] == 1000:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        print("load pretrained")
        model.load_state_dict(checkpoint["model"])
    elif pretrained and kwargs['num_classes'] == 100:
        raise ValueError('No trained model provided')
    return model

