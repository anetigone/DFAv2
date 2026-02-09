"""
FMD: Frequency-Modulated Denoiser
频率调制恢复器

深度先验网络，使用频率信息动态调制
包含频率门控注意力(FGA)和动态参数调制(AdaFM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LayerNorm2d, FrequencyGatedAttention, AdaptiveFrequencyModulation, CBAM


class ResidualBlock(nn.Module):
    """基础残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.act(self.conv1(out))
        out = self.norm2(out)
        out = self.conv2(out)
        return identity + out * self.res_scale


class FrequencyModulatedBlock(nn.Module):
    """
    频率调制块

    结合频率门控注意力和自适应频率调制
    """
    def __init__(self, channels, freq_dim=64):
        super(FrequencyModulatedBlock, self).__init__()
        self.channels = channels
        self.freq_dim = freq_dim

        # 频率门控注意力
        self.fga = FrequencyGatedAttention(channels)

        # 自适应频率调制
        self.adafm = AdaptiveFrequencyModulation(channels, freq_dim)

        # 通道和空间注意力
        self.cbam = CBAM(channels)

        # 主干卷积
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # 残差缩放
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, freq_mask, freq_feat):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            freq_mask: 频率掩码 [B, 1, H, W//2+1]
            freq_feat: 频率特征 [B, freq_dim]
        Returns:
            out: 输出特征 [B, C, H, W]
        """
        identity = x

        # 1. 频率门控注意力
        x = self.fga(x, freq_mask)

        # 2. 自适应频率调制
        x = self.adafm(x, freq_feat)

        # 3. CBAM 注意力
        x = self.cbam(x)

        # 4. 主干处理
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.conv2(out)

        # 残差连接
        out = identity + out * self.res_scale

        return out


class FMD(nn.Module):
    """
    频率调制恢复器 (Frequency-Modulated Denoiser)

    深度先验网络，实现 HQS 的先验项子问题
    x_k = Denoiser(z_k, sigma_k)
    """
    def __init__(self, in_channels=3, feat_channels=64, num_blocks=6, freq_dim=64):
        super(FMD, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        # 头部特征提取
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            LayerNorm2d(feat_channels)
        )

        # 频率调制块
        self.fm_blocks = nn.ModuleList([
            FrequencyModulatedBlock(feat_channels, freq_dim)
            for _ in range(num_blocks)
        ])

        # 全局特征融合
        self.global_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_channels, freq_dim, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 尾部重建
        self.tail = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels // 2, in_channels, 3, 1, 1),
        )

        # 全局残差缩放
        self.global_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, z_k, params):
        """
        Args:
            z_k: DFU 的输出 [B, 3, H, W]
            params: DFAP 的参数字典，包含:
                - freq_mask: 频率掩码 [B, 1, H, W//2+1]
                - freq_feat: 频率特征 [B, freq_dim]
        Returns:
            x_k: 恢复结果 [B, 3, H, W]
        """
        # 提取参数
        freq_mask = params['freq_mask']
        freq_feat = params['freq_feat']
        B = z_k.shape[0]

        # 初始特征提取
        feat = self.head(z_k)  # [B, feat_channels, H, W]

        # 全局特征提取
        global_feat = self.global_fusion(feat).view(B, -1)  # [B, freq_dim]

        # 融合频率特征
        fused_freq_feat = (freq_feat + global_feat) / 2.0

        # 通过频率调制块
        for fm_block in self.fm_blocks:
            feat = fm_block(feat, freq_mask, fused_freq_feat)

        # 重建
        residual = self.tail(feat)  # [B, 3, H, W]
        residual = F.tanh(residual) # 限制在 [-1, 1]

        # 全局残差连接
        x_k = z_k + residual * self.global_scale

        # 限制输出范围
        x_k = torch.clamp(x_k, min=-1.0, max=1.0)

        return x_k


class SimpleFMD(nn.Module):
    """
    简化版 FMD，用于快速实验

    使用标准残差块 + 通道注意力
    """
    def __init__(self, in_channels=3, feat_channels=64, num_blocks=4):
        super(SimpleFMD, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        # 头部
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 主体
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
                LayerNorm2d(feat_channels)
            )
            for _ in range(num_blocks)
        ])

        # 尾部
        self.tail = nn.Conv2d(feat_channels, in_channels, 3, 1, 1)

        # 缩放因子
        self.global_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, z_k, params):
        """
        Args:
            z_k: DFU 输出 [B, 3, H, W]
            params: 参数字典 (不使用)
        Returns:
            x_k: 恢复结果 [B, 3, H, W]
        """
        # 头部
        feat = self.head(z_k)

        # 主体
        for block in self.body:
            feat = feat + block(feat) * 0.1

        # 尾部
        residual = self.tail(feat)

        # 全局残差
        x_k = z_k + residual * self.global_scale

        return x_k
