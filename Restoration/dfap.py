"""
DFAP: Dynamic Frequency-Aware Parameter Estimator
动态频率感知参数估计器

负责动态估计退化模型参数: K(模糊核), T(增益), D(偏置), 以及频率掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .layers import LayerNorm2d, AdaptiveFrequencyModulation


class DFAP(nn.Module):
    """
    动态频率感知参数估计器

    输入: 原始退化图 y, 上一阶段恢复图 x_{k-1}
    输出:
        - K_k: 模糊核图 [B, 1, kernel_size, kernel_size]
        - T_k: 增益图 [B, 1, H, W]
        - D_k: 偏置图 [B, 1, H, W]
        - rho_k: 超参数 [B, 1]
        - freq_mask: 频率掩码 [B, 1, H, W//2+1]
    """
    def __init__(self, in_channels=3, kernel_size=15, num_freq_bands=8, feat_dim=64):
        super(DFAP, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_freq_bands = num_freq_bands
        self.feat_dim = feat_dim

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 下采样
            nn.LeakyReLU(0.2, inplace=True),
            LayerNorm2d(64),
            nn.Conv2d(64, feat_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            LayerNorm2d(feat_dim)
        )

        # 频率分析头
        self.freq_analyzer = FrequencyAnalyzer(in_channels, num_freq_bands, feat_dim)

        # 参数预测头
        self.kernel_predictor = KernelPredictor(feat_dim, kernel_size)
        self.gain_predictor = GainPredictor(feat_dim)
        self.bias_predictor = BiasPredictor(feat_dim)
        self.rho_predictor = RhoPredictor(feat_dim)
        self.freq_mask_predictor = FreqMaskPredictor(feat_dim, num_freq_bands)

    def forward(self, y, x_prev=None):
        """
        Args:
            y: 原始退化图 [B, 3, H, W]
            x_prev: 上一阶段恢复图 [B, 3, H, W] (可选)
        Returns:
            params: dict包含 K, T, D, rho, freq_mask
        """
        B, _, H, W = y.shape

        # 如果没有上一阶段输出，使用退化图初始化
        if x_prev is None:
            x_prev = y

        # 拼接输入
        x_concat = torch.cat([y, x_prev], dim=1)  # [B, 6, H, W]

        # 提取空间特征
        spatial_feat = self.feature_extractor(x_concat)  # [B, feat_dim, H/2, W/2]

        # 全局池化
        global_feat = F.adaptive_avg_pool2d(spatial_feat, 1).view(B, -1)  # [B, feat_dim]

        # 频率分析
        freq_feat = self.freq_analyzer(y)  # [B, feat_dim]

        # 融合空间和频率特征
        combined_feat = (spatial_feat.mean(dim=[2, 3]) + freq_feat + global_feat) / 3.0

        # 预测各个参数
        K = self.kernel_predictor(combined_feat)  # [B, 1, kH, kW]
        T = self.gain_predictor(spatial_feat, H, W)  # [B, 1, H, W]
        D = self.bias_predictor(spatial_feat, H, W)  # [B, 1, H, W]
        rho = self.rho_predictor(combined_feat)  # [B, 1]
        freq_mask = self.freq_mask_predictor(freq_feat, H, W)  # [B, 1, H, W//2+1]

        params = {
            'K': K,           # 模糊核
            'T': T,           # 增益
            'D': D,           # 偏置
            'rho': rho,       # 超参数
            'freq_mask': freq_mask,  # 频率掩码
            'spatial_feat': spatial_feat,  # 空间特征(供后续模块使用)
            'freq_feat': freq_feat        # 频率特征
        }

        return params


class FrequencyAnalyzer(nn.Module):
    """频率分析器：分析图像的频域特征"""
    def __init__(self, in_channels, num_bands, feat_dim):
        super(FrequencyAnalyzer, self).__init__()
        self.num_bands = num_bands
        self.in_channels = in_channels

        # 每个频段提取统计量：mean, max, std
        freq_input_dim = in_channels * num_bands * 3

        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_input_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, feat_dim]
        """
        B, C, H, W = x.shape

        # 转换到频域
        x_fft = torch.fft.rfft2(x.float(), norm='ortho')
        mag = torch.abs(x_fft)  # [B, C, H, W//2+1]

        # 生成频率半径坐标
        u = torch.fft.rfftfreq(W).to(x.device)
        v = torch.fft.fftfreq(H).to(x.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        radius = torch.sqrt(u_grid**2 + v_grid**2)

        max_r = radius.max()

        # 对每个频段提取统计量
        band_features = []
        for i in range(self.num_bands):
            lower = max_r * (i / self.num_bands)
            upper = max_r * ((i + 1) / self.num_bands)
            mask = ((radius >= lower) & (radius < upper)).float()
            mask_sum = mask.sum() + 1e-8

            # 应用掩码
            m_mag = mag * mask

            # 计算统计量
            mean_val = m_mag.sum(dim=(2, 3)) / (mask_sum)  # [B, C]
            max_val = m_mag.view(B, C, -1).amax(dim=2)  # [B, C]
            # 标准差
            std_val = torch.sqrt(((m_mag - mean_val.view(B, C, 1, 1))**2 * mask).sum(dim=(2, 3)) / mask_sum)

            band_features.extend([mean_val, max_val, std_val])

        # 拼接所有频段特征
        feat_vec = torch.cat(band_features, dim=1)  # [B, C * num_bands * 3]

        # 通过MLP提取高层特征
        feat = self.freq_mlp(feat_vec)  # [B, feat_dim]

        return feat


class KernelPredictor(nn.Module):
    """模糊核预测器"""
    def __init__(self, feat_dim, kernel_size):
        super(KernelPredictor, self).__init__()
        self.kernel_size = kernel_size

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, kernel_size * kernel_size),
            nn.Softmax(dim=1)  # 归一化，和为1
        )

    def forward(self, feat):
        """
        feat: [B, feat_dim]
        Returns: [B, 1, kH, kW]
        """
        B = feat.shape[0]
        kernel = self.mlp(feat)
        kernel = kernel.view(B, 1, self.kernel_size, self.kernel_size)
        return kernel


class GainPredictor(nn.Module):
    """增益图预测器 (T)"""
    def __init__(self, feat_dim, out_channels=1):
        super(GainPredictor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim // 2, out_channels, 3, 1, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]，后续可以缩放
        )

    def forward(self, feat, H, W):
        """
        feat: [B, feat_dim, h, w]
        Returns: [B, 1, H, W] 上采样到原始尺寸
        """
        gain = self.conv(feat)
        # 上采样
        gain = F.interpolate(gain, size=(H, W), mode='bilinear', align_corners=False)
        # 缩放到合理范围
        gain = torch.exp(gain * 2.0 - 1.0)  # [exp(-1), exp(1)] ≈ [0.37, 2.72]
        return gain


class BiasPredictor(nn.Module):
    """偏置图预测器 (D)"""
    def __init__(self, feat_dim, out_channels=3):
        super(BiasPredictor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim // 2, out_channels, 3, 1, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, feat, H, W):
        """
        feat: [B, feat_dim, h, w]
        Returns: [B, 1, H, W]
        """
        bias = self.conv(feat)
        bias = F.interpolate(bias, size=(H, W), mode='bilinear', align_corners=False)
        # 缩放到合理范围 [-0.5, 0.5]
        bias = bias * 0.5
        return bias


class RhoPredictor(nn.Module):
    """超参数 rho 预测器"""
    def __init__(self, feat_dim):
        super(RhoPredictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim // 2, 1),
            nn.Softplus()  # 输出正数
        )

    def forward(self, feat):
        """
        feat: [B, feat_dim]
        Returns: [B, 1]
        """
        rho = self.mlp(feat)
        # 限制范围 [0.1, 1.0]
        rho = F.sigmoid(rho) * 0.9 + 0.1
        return rho


class FreqMaskPredictor(nn.Module):
    """频率掩码预测器"""
    def __init__(self, feat_dim, num_bands):
        super(FreqMaskPredictor, self).__init__()
        self.num_bands = num_bands

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, num_bands),
            nn.Softmax(dim=1)  # 转换为权重分布
        )

    def forward(self, freq_feat, H, W):
        """
        freq_feat: [B, feat_dim]
        Returns: [B, 1, H, W//2+1] 频率掩码
        """
        B = freq_feat.shape[0]

        # 预测每个频段的权重
        band_weights = self.mlp(freq_feat)  # [B, num_bands]

        # 生成频率坐标
        u = torch.fft.rfftfreq(W).to(freq_feat.device)
        v = torch.fft.fftfreq(H).to(freq_feat.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        radius = torch.sqrt(u_grid**2 + v_grid**2)

        max_r = radius.max()

        # 根据频段权重生成掩码
        freq_mask = torch.zeros(B, 1, H, W // 2 + 1, device=freq_feat.device)

        for i in range(self.num_bands):
            lower = max_r * (i / self.num_bands)
            upper = max_r * ((i + 1) / self.num_bands)
            mask = ((radius >= lower) & (radius < upper)).float()

            # 应用权重
            freq_mask = freq_mask + mask.unsqueeze(0).unsqueeze(0) * band_weights[:, i:i+1].view(B, 1, 1, 1)

        return freq_mask
