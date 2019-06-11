# -*- coding: utf-8 -*-

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LatentBlock"]


def do_weight_init_(module):
    assert isinstance(module, nn.Module)
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, std=1e-2)


class LatentKernel(nn.Module):

    def __init__(self,
                 vis_channels,
                 latent_channels,
                 norm_layer,
                 non_linear_layer,
                 norm_func,
                 mode,
                 graph_conv):
        super(LatentKernel, self).__init__()

        self.vis_channels = vis_channels
        self.latent_channels = latent_channels
        self.norm_layer = norm_layer
        self.non_linear_layer = non_linear_layer
        self.norm_func = norm_func
        self.mode = mode
        self.graph_conv = graph_conv

        if self.mode == "symmetric":
            self.psi = nn.Sequential(
                nn.Conv2d(vis_channels, latent_channels, kernel_size=1, bias=False),
                self.norm_layer(latent_channels),
                self.non_linear_layer(inplace=True),
            )
        elif self.mode == "asymmetric":
            self.psi_v2l = nn.Sequential(
                nn.Conv2d(vis_channels, latent_channels, kernel_size=1, bias=False),
                self.norm_layer(latent_channels),
                self.non_linear_layer(inplace=True),
            )
            self.psi_l2v = nn.Sequential(
                nn.Conv2d(vis_channels, latent_channels, kernel_size=1, bias=False),
                self.norm_layer(latent_channels),
                self.non_linear_layer(inplace=True),
            )
        else:
            raise ValueError(f"unsupported latent mode: {self.mode}")

        if self.graph_conv:
            self.graph_conv_mapping = nn.Sequential(
                nn.Linear(vis_channels, vis_channels, bias=False),
                self.non_linear_layer(inplace=True)
            )
            self.graph_conv_nonlinear = self.non_linear_layer(inplace=True)

    def forward(self, v2l_conv_feat, l2v_conv_feat):
        n, c_in, h_in, w_in = v2l_conv_feat.shape

        if self.mode == "symmetric":
            assert l2v_conv_feat is None
            c_out, h_out, w_out = c_in, h_in, w_in
            graph_adj = self.psi(v2l_conv_feat)
            v2l_graph_adj = l2v_graph_adj = self.norm_func(graph_adj.reshape(n, -1, h_in * w_in), dim=1)
        else:
            _, c_out, h_out, w_out = l2v_conv_feat.shape
            v2l_graph_adj = self.psi_v2l(v2l_conv_feat)
            v2l_graph_adj = self.norm_func(v2l_graph_adj.reshape(n, -1, h_in * w_in), dim=2)  # ???
            l2v_graph_adj = self.psi_l2v(l2v_conv_feat)
            l2v_graph_adj = self.norm_func(l2v_graph_adj.reshape(n, -1, h_out * w_out), dim=1)

        # project to latent space
        latent_node_feat = torch.bmm(v2l_graph_adj,
                                     v2l_conv_feat.reshape(n, -1, h_in * w_in).permute(0, 2, 1))
        if self.graph_conv:
            latent_node_feat = self.graph_conv_nonlinear(latent_node_feat)

        # internal projection within latent space
        latent_node_feat_normed = self.norm_func(latent_node_feat, dim=-1)
        affinity_mapping = torch.bmm(latent_node_feat_normed,
                                     latent_node_feat_normed.permute(0, 2, 1))
        affinity_mapping = torch.softmax(affinity_mapping, dim=-1)
        latent_node_feat = torch.bmm(affinity_mapping, latent_node_feat)
        if self.graph_conv:
            latent_node_feat = self.graph_conv_mapping(latent_node_feat.reshape(-1, c_in)).reshape(n, -1, c_in)

        # project back to visible space
        vis_feat = torch.bmm(latent_node_feat.permute(0, 2, 1), l2v_graph_adj).reshape(n, -1, h_out, w_out)
        if self.graph_conv:
            vis_feat = self.graph_conv_nonlinear(vis_feat)

        return vis_feat


class LatentBlock(nn.Module):

    def __init__(self,
                 vis_channels,
                 latent_channels,
                 channel_shrink=4,
                 upsample_ratio=1,
                 num_kernels=1,
                 latent_proj_mode="asymmetric",
                 upsample_mode="bilinear",
                 use_residual=True,
                 norm_layer=nn.BatchNorm2d,
                 non_linear_layer=nn.ReLU,
                 norm_func=F.normalize,
                 graph_conv=True):
        super(LatentBlock, self).__init__()

        if isinstance(latent_channels, int):
            latent_channels = (latent_channels, )
        assert len(latent_channels) == num_kernels
        assert latent_proj_mode in ("asymmetric", "symmetric")

        self.vis_channels = vis_channels
        self.latent_channels = latent_channels
        self.channel_shrink = channel_shrink
        self.upsample_ratio = upsample_ratio
        self.num_kernels = num_kernels
        self.latent_proj_mode = latent_proj_mode
        self.upsample_mode = upsample_mode
        self.use_residual = use_residual
        self.norm_layer = norm_layer
        self.non_linear_layer = non_linear_layer
        self.norm_func = norm_func
        self.graph_conv = graph_conv

        pre_latent_channels = self.vis_channels // self.channel_shrink
        if self.latent_proj_mode == "symmetric":
            assert self.upsample_ratio == 1
            self.channel_shrink_mapping = nn.Sequential(
                nn.Conv2d(vis_channels, pre_latent_channels, kernel_size=1, bias=False),
                self.norm_layer(pre_latent_channels),
            )
        else:
            self.channel_shrink_mapping_v2l = nn.Sequential(
                nn.Conv2d(vis_channels, pre_latent_channels, kernel_size=1, bias=False),
                self.norm_layer(pre_latent_channels),
            )

            if self.upsample_ratio == 1:
                self.channel_shrink_mapping_l2v = nn.Sequential(
                    nn.Conv2d(vis_channels, pre_latent_channels, kernel_size=1, bias=False),
                    self.norm_layer(pre_latent_channels),
                )
            elif self.upsample_mode in ("transconv", "deconv"):
                self.channel_shrink_mapping_l2v = nn.Sequential(
                    nn.ConvTranspose2d(vis_channels,
                                       pre_latent_channels,
                                       kernel_size=2 * self.upsample_ratio - self.upsample_ratio % 2,
                                       padding=ceil((self.upsample_ratio - 1.) / 2.),
                                       stride=self.upsample_ratio,
                                       bias=False),
                    self.norm_layer(pre_latent_channels),
                )
            else:
                self.channel_shrink_mapping_l2v = nn.Sequential(
                    nn.Conv2d(vis_channels, pre_latent_channels, kernel_size=1, bias=False),
                    nn.Upsample(scale_factor=self.upsample_ratio,
                                mode=self.upsample_mode,
                                align_corners=False if "linear" in self.upsample_mode else None),
                    self.norm_layer(pre_latent_channels),
                )

        for i, latent_channel in enumerate(self.latent_channels):
            self.add_module(f"latent_kernel_{i}",
                            LatentKernel(pre_latent_channels,
                                         latent_channel,
                                         self.norm_layer,
                                         self.non_linear_layer,
                                         self.norm_func,
                                         self.latent_proj_mode,
                                         self.graph_conv))

        self.channel_expand_mapping = nn.Sequential(
            nn.Conv2d(pre_latent_channels * self.num_kernels, vis_channels, kernel_size=1, bias=False),
            self.norm_layer(vis_channels),
        )

        if self.use_residual:
            self.gamma = nn.Parameter(torch.tensor(0.0))

        do_weight_init_(self)

    def forward(self, x):
        if self.latent_proj_mode == "symmetric":
            v2l_feat = self.channel_shrink_mapping(x)
            v2l_feat = self.norm_func(v2l_feat, dim=1)
            l2v_feat = None
        else:
            v2l_feat = self.channel_shrink_mapping_v2l(x)
            v2l_feat = self.norm_func(v2l_feat, dim=1)
            l2v_feat = self.channel_shrink_mapping_l2v(x)
            l2v_feat = self.norm_func(l2v_feat, dim=1)

        out_feats = []
        for i in range(self.num_kernels):
            latent_feat = self._modules[f"latent_kernel_{i}"](v2l_feat, l2v_feat)
            out_feats.append(latent_feat)

        out_feat = torch.cat(out_feats, dim=1)
        out_feat = self.channel_expand_mapping(out_feat)

        if self.use_residual:
            return x + out_feat * self.gamma
        else:
            return out_feat
