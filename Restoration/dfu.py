"""
DFU: Data Fidelity Unit
数据保真单元

实现 HQS (Half Quadratic Splitting) 的数据保真项子问题的解析解
解析解公式: z_k = [T ⊙ (K^T ⊗ (y - D)) + rho * x_{k-1}] / [T^2 ⊙ (K^T ⊗ K) + rho]
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from .layers import safe_divide


class DFU(nn.Module):
    """
    数据保真单元 (Data Fidelity Unit)

    不含可学习参数，直接实现 HQS 解析解公式
    强制恢复图符合物理过程: y = K ⊗ (x ⊙ T) + D
    """
    def __init__(self):
        super(DFU, self).__init__()

    def forward(self, y, x_prev, params):
        """
        Args:
            y: 原始退化图 [B, C, H, W]
            x_prev: 上一阶段输出 [B, C, H, W]
            params: DFAP 估计的参数字典，包含:
                - K: 模糊核 [B, 1, kH, kW]
                - T: 增益图 [B, 1, H, W]
                - D: 偏置图 [B, 1, H, W]
                - rho: 超参数 [B, 1]
        Returns:
            z_k: 数据保真后的结果 [B, C, H, W]
        """
        B, C, H, W = y.shape
        K = params['K']       # [B, 1, kH, kW]
        T = params['T']       # [B, 1, H, W]
        D = params['D']       # [B, 1, H, W]
        rho = params['rho']   # [B, 1]

        # 扩展维度
        rho = rho.view(B, 1, 1, 1)
        if T.shape[1] == 1: T = T.expand(B, C, H, W)
        if D.shape[1] == 1: D = D.expand(B, C, H, W)

        # 1. 预计算频域核 F(K)
        # s=(H, W) 会自动处理补零填充 (Zero-padding)
        F_K = torch.fft.rfft2(K, s=(H, W), norm='ortho')
        F_K_conj = torch.conj(F_K)

        # 2. 计算分子项: T ⊙ (K^T ⊗ (y - D))
        y_minus_D = y - D
        F_y_minus_D = torch.fft.rfft2(y_minus_D, norm='ortho')
        # K^T ⊗ 在频域就是乘共轭
        K_trans_y = torch.fft.irfft2(F_K_conj * F_y_minus_D, s=(H, W), norm='ortho')
        numerator = T * K_trans_y + rho * x_prev

        # 3. 计算分母项: T^2 ⊙ (K^T ⊗ K ⊗ 1) + rho
        # K^T ⊗ K 在频域是 |F(K)|^2
        F_KK = F_K_conj * F_K
        K_energy = torch.fft.irfft2(F_KK, s=(H, W), norm='ortho')
        # 这里的 K_energy 描述了退化算子的空间增益分布
        denominator = (T ** 2) * K_energy + rho

        # 4. 安全除法
        z_k = numerator / (denominator + 1e-8)

        # 5. 范围约束 (针对图像恢复，通常在 -1 到 1 或 0 到 1)
        return torch.clamp(z_k, min=-1.0, max=1.0)
