"""
DFA-DUN: 基础层和工具函数
包含: LayerNorm, 注意力机制, 频域工具等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class LayerNorm2d(nn.Module):
    """2D Layer Normalization for images"""
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True)
        res = (x - mean) / (std + self.eps)
        res = res * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return res


class FrequencyGatedAttention(nn.Module):
    """
    频率门控注意力 (Frequency Gated Attention - FGA)
    在频域应用动态掩码进行特征调制
    """
    def __init__(self, channels):
        super(FrequencyGatedAttention, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 1)
        self.norm = LayerNorm2d(channels)

    def forward(self, x, freq_mask):
        """
        x: 输入特征 [B, C, H, W]
        freq_mask: 频率掩码 [B, 1, H, W//2+1] (rfft后的形状)
        """
        B, C, H, W = x.shape

        # 转换到频域
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # 应用频率掩码 (门控)
        # freq_mask 需要匹配频域维度
        if freq_mask.size(-1) != x_freq.size(-1):
            # 调整掩码大小
            freq_mask = F.interpolate(
                freq_mask.float(),
                size=(H, W // 2 + 1),
                mode='bilinear',
                align_corners=False
            )

        # 门控: 实部和虚部都乘以掩码
        x_freq_real = x_freq.real * freq_mask
        x_freq_imag = x_freq.imag * freq_mask
        x_freq_mod = torch.complex(x_freq_real, x_freq_imag)

        # 转换回空间域
        x_spatial = torch.fft.irfft2(x_freq_mod, norm='ortho', s=(H, W))

        # 残差连接和归一化
        out = self.norm(x + x_spatial)
        out = self.conv(out)

        return out


class AdaptiveFrequencyModulation(nn.Module):
    """
    自适应频率调制 (AdaFM)
    类似 StyleGAN 的 SPADE，用于动态调制网络特征
    """
    def __init__(self, in_channels, freq_dim=64):
        super(AdaptiveFrequencyModulation, self).__init__()
        self.in_channels = in_channels
        self.freq_dim = freq_dim

        # 将频率特征映射到 scale 和 bias
        self.fc = nn.Sequential(
            nn.Linear(freq_dim, in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 初始化为接近恒等映射
        nn.init.zeros_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)

    def forward(self, x, freq_feat):
        """
        x: 输入特征 [B, C, H, W]
        freq_feat: 频率特征向量 [B, freq_dim]
        """
        B, C, H, W = x.shape

        # 生成调制参数
        params = self.fc(freq_feat)  # [B, C*2]
        params = params.view(B, 2, C, 1, 1)

        scale = params[:, 0] + 1.0  # [B, C, 1, 1], 初始化为1
        bias = params[:, 1]          # [B, C, 1, 1], 初始化为0

        # 应用调制
        out = x * scale + bias

        return out


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


def safe_divide(numerator, denominator, eps=1e-8):
    """安全除法，避免除零"""
    return numerator / (denominator.abs() + eps)


def fft_convolve(input, kernel):
    """
    使用FFT进行快速卷积
    input: [B, C, H, W]
    kernel: [B, 1, kH, kW] 或 [1, 1, kH, kW]
    """
    B, C, H, W = input.shape
    kH, kW = kernel.shape[-2:]

    # 计算填充大小
    pad_h = kH // 2
    pad_w = kW // 2

    # 填充输入
    input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # FFT卷积
    input_fft = torch.fft.rfft2(input_padded, norm='ortho')
    kernel_fft = torch.fft.rfft2(kernel, s=(input_padded.shape[-2], input_padded.shape[-1]), norm='ortho')

    output_fft = input_fft * kernel_fft.conj()
    output = torch.fft.irfft2(output_fft, norm='ortho')

    # 裁剪回原始大小
    output = output[:, :, :H, :W]

    return output
