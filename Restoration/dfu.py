"""
DFU: Data Fidelity Unit
数据保真单元

实现 HQS (Half Quadratic Splitting) 的数据保真项子问题的解析解
解析解公式: z_k = [T ⊙ (K^T ⊗ (y - D)) + rho * x_{k-1}] / [T^2 ⊙ (K^T ⊗ K) + rho]
"""

import torch
import torch.nn as nn
import torch.fft
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

        # 提取参数
        K = params['K']       # [B, 1, kH, kW]
        T = params['T']       # [B, 1, H, W]
        D = params['D']       # [B, 1, H, W]
        rho = params['rho']   # [B, 1]

        # 扩展 T 和 D 到所有通道
        if T.shape[1] == 1:
            T = T.expand(B, C, H, W)
        if D.shape[1] == 1:
            D = D.expand(B, C, H, W)

        # 限制范围避免数值不稳定
        T = torch.clamp(T, min=0.1, max=5.0)
        D = torch.clamp(D, min=-1.0, max=1.0)

        # 计算解析解
        z_k = self._hqs_analytical_solution(y, x_prev, K, T, D, rho)

        return z_k

    def _hqs_analytical_solution(self, y, x_prev, K, T, D, rho):
        """
        计算 HQS 解析解:
        z_k = [T ⊙ (K^T ⊗ (y - D)) + rho * x_{k-1}] / [T^2 ⊙ (K^T ⊗ K) + rho]

        使用 FFT 加速卷积运算
        """
        B, C, H, W = y.shape
        device = y.device

        # 1. 计算 y - D
        y_minus_D = y - D  # [B, C, H, W]

        # 2. 计算 K^T ⊗ (y - D) (转置卷积)
        # 在频域中，转置卷积等于卷积的复共轭
        K_trans_conv = self._fft_conv_transpose(y_minus_D, K)  # [B, C, H, W]

        # 3. 计算分子第一项: T ⊙ (K^T ⊗ (y - D))
        term1_numerator = T * K_trans_conv  # [B, C, H, W]

        # 4. 计算分子第二项: rho * x_{k-1}
        rho_expanded = rho.view(B, 1, 1, 1)
        term2_numerator = rho_expanded * x_prev  # [B, C, H, W]

        # 5. 计算分子: T ⊙ (K^T ⊗ (y - D)) + rho * x_{k-1}
        numerator = term1_numerator + term2_numerator  # [B, C, H, W]

        # 6. 计算分母第一项: K^T ⊗ K
        # 创建一个全1的特征图来进行 K ⊗ K
        ones_map = torch.ones(B, C, H, W, device=device)
        K_self_conv = self._fft_conv(ones_map, K)  # [B, C, H, W]

        # 7. 计算分母: T^2 ⊙ (K^T ⊗ K) + rho
        T_squared = T ** 2
        denominator = T_squared * K_self_conv + rho_expanded  # [B, C, H, W]

        # 8. 安全除法
        z_k = safe_divide(numerator, denominator, eps=1e-8)

        # 9. 限制输出范围
        z_k = torch.clamp(z_k, min=-1.0, max=1.0)

        return z_k

    def _fft_conv(self, x, kernel):
        """
        使用 FFT 进行卷积
        x: [B, C, H, W]
        kernel: [B, 1, kH, kW]
        Returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        kH, kW = kernel.shape[-2:]

        # 填充
        pad_h = kH // 2
        pad_w = kW // 2
        x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        # FFT
        x_fft = torch.fft.rfft2(x_padded.float(), norm='ortho')
        kernel_fft = torch.fft.rfft2(kernel.float(), s=(x_padded.shape[-2], x_padded.shape[-1]), norm='ortho')

        # 频域相乘
        output_fft = x_fft * kernel_fft.conj()

        # IFFT
        output = torch.fft.irfft2(output_fft, norm='ortho')

        # 裁剪
        output = output[:, :, :H, :W]

        return output.type_as(x)

    def _fft_conv_transpose(self, x, kernel):
        """
        使用 FFT 进行转置卷积 (相关操作)
        x: [B, C, H, W]
        kernel: [B, 1, kH, kW]
        Returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        kH, kW = kernel.shape[-2:]

        # 翻转 kernel (转置卷积 = 翻转后的卷积)
        kernel_flipped = torch.flip(kernel, dims=[-2, -1])

        # 填充
        pad_h = kH // 2
        pad_w = kW // 2
        x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        # FFT
        x_fft = torch.fft.rfft2(x_padded.float(), norm='ortho')
        kernel_fft = torch.fft.rfft2(kernel_flipped.float(), s=(x_padded.shape[-2], x_padded.shape[-1]), norm='ortho')

        # 频域相乘
        output_fft = x_fft * kernel_fft.conj()

        # IFFT
        output = torch.fft.irfft2(output_fft, norm='ortho')

        # 裁剪
        output = output[:, :, :H, :W]

        return output.type_as(x)
