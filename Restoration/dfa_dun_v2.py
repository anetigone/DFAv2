"""
DFA-DUN: Dynamic Frequency-Aware Deep Unfolding Network
动态频率感知深度展开网络

基于 HQS (Half Quadratic Splitting) 的 All-in-One 图像恢复模型
统一物理模型: y = K ⊗ (x ⊙ T) + D

Author: DFA-DUN Team
"""

import torch
import torch.nn as nn
from .dfap import DFAP
from .dfu import DFU
from .fmd import FMD, SimpleFMD


class DFADUN(nn.Module):
    """
    DFA-DUN: 动态频率感知深度展开网络

    由 S 个阶段级联而成，每个阶段包含:
    1. DFAP: 动态频率感知参数估计器
    2. DFU: 数据保真单元 (HQS解析解)
    3. FMD: 频率调制恢复器 (深度先验)

    支持任务: 去雨、去噪、去雾、去模糊、低光照增强等
    """
    def __init__(
        self,
        in_channels=3,
        num_stages=4,
        kernel_size=15,
        num_freq_bands=8,
        feat_dim=64,
        num_fmd_blocks=6,
        use_simple_fmd=False
    ):
        """
        Args:
            in_channels: 输入通道数 (RGB=3)
            num_stages: 展开阶段数
            kernel_size: 预测模糊核的大小
            num_freq_bands: 频率分析带数
            feat_dim: 特征维度
            num_fmd_blocks: FMD 中的模块数
            use_simple_fmd: 是否使用简化版 FMD (用于快速实验)
        """
        super(DFADUN, self).__init__()
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.use_simple_fmd = use_simple_fmd

        # 为每个阶段创建模块
        self.dfap_list = nn.ModuleList()
        self.dfu = DFU()  # DFU 不含参数，可以共享
        self.fmd_list = nn.ModuleList()

        for stage_idx in range(num_stages):
            # DFAP: 参数估计器
            dfap = DFAP(
                in_channels=in_channels,
                kernel_size=kernel_size,
                num_freq_bands=num_freq_bands,
                feat_dim=feat_dim
            )
            self.dfap_list.append(dfap)

            # FMD: 频率调制恢复器
            if use_simple_fmd:
                fmd = SimpleFMD(
                    in_channels=in_channels,
                    feat_channels=feat_dim,
                    num_blocks=num_fmd_blocks // 2  # 简化版使用较少块
                )
            else:
                fmd = FMD(
                    in_channels=in_channels,
                    feat_channels=feat_dim,
                    num_blocks=num_fmd_blocks,
                    freq_dim=feat_dim
                )
            self.fmd_list.append(fmd)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                # 保留缩放参数的初始值
                pass

    def forward(self, y, return_params=False):
        """
        前向传播

        Args:
            y: 退化图像 [B, 3, H, W]
            return_params: 是否返回中间参数 (用于可视化或分析)

        Returns:
            x: 恢复图像 [B, 3, H, W]
            all_params: (可选) 所有阶段的参数列表
        """
        # 初始化
        x = y  # 初始恢复图 = 退化图
        all_params = []

        # 逐阶段处理
        for stage_idx in range(self.num_stages):
            # 1. DFAP: 估计退化参数
            params = self.dfap_list[stage_idx](y, x)
            all_params.append(params)

            # 2. DFU: 数据保真 (HQS解析解)
            z_k = self.dfu(y, x, params)

            # 3. FMD: 频率调制深度恢复
            x = self.fmd_list[stage_idx](z_k, params)

        if return_params:
            return x, all_params
        else:
            return x

    def get_stage_output(self, y, stage_idx):
        """
        获取指定阶段的输出 (用于调试或中间结果可视化)

        Args:
            y: 退化图像 [B, 3, H, W]
            stage_idx: 阶段索引 (0 到 num_stages-1)

        Returns:
            x_stage: 该阶段的输出 [B, 3, H, W]
        """
        x = y

        for i in range(min(stage_idx + 1, self.num_stages)):
            params = self.dfap_list[i](y, x)
            z_k = self.dfu(y, x, params)
            x = self.fmd_list[i](z_k, params)

        return x

    def estimate_degradation(self, y):
        """
        仅使用第一个 DFAP 估计退化类型和参数

        Args:
            y: 退化图像 [B, 3, H, W]

        Returns:
            params: 第一个阶段的参数估计
        """
        with torch.no_grad():
            params = self.dfap_list[0](y, y)
        return params


class DFADUNLite(nn.Module):
    """
    DFA-DUN 轻量版

    使用更少的阶段和更小的网络，适合实时应用
    """
    def __init__(
        self,
        in_channels=3,
        num_stages=2,
        kernel_size=11,
        num_freq_bands=4,
        feat_dim=32,
        num_fmd_blocks=3
    ):
        super(DFADUNLite, self).__init__()
        self.in_channels = in_channels
        self.num_stages = num_stages

        # DFAP 列表
        self.dfap_list = nn.ModuleList([
            DFAP(
                in_channels=in_channels,
                kernel_size=kernel_size,
                num_freq_bands=num_freq_bands,
                feat_dim=feat_dim
            )
            for _ in range(num_stages)
        ])

        # DFU (共享)
        self.dfu = DFU()

        # FMD 列表 (使用简化版)
        self.fmd_list = nn.ModuleList([
            SimpleFMD(
                in_channels=in_channels,
                feat_channels=feat_dim,
                num_blocks=num_fmd_blocks
            )
            for _ in range(num_stages)
        ])

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, y):
        """
        前向传播

        Args:
            y: 退化图像 [B, 3, H, W]

        Returns:
            x: 恢复图像 [B, 3, H, W]
        """
        x = y

        for stage_idx in range(self.num_stages):
            # DFAP
            params = self.dfap_list[stage_idx](y, x)

            # DFU
            z_k = self.dfu(y, x, params)

            # FMD
            x = self.fmd_list[stage_idx](z_k, params)

        return x


# 模型尺寸对比函数
def get_model_flops(model, input_size=(3, 256, 256)):
    """
    估算模型的 FLOPs (需要安装 thop 库)

    pip install thop
    """
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(1, *input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        return "N/A (需要安装 thop)"


def build_dfa_dun(model_type='standard', **kwargs):
    """
    构建 DFA-DUN 模型的工厂函数

    Args:
        model_type: 模型类型
            - 'standard': 标准版 DFADUN
            - 'lite': 轻量版 DFADUNLite
        **kwargs: 模型参数

    Returns:
        model: DFA-DUN 模型
    """
    if model_type == 'standard':
        model = DFADUN(**kwargs)
    elif model_type == 'lite':
        model = DFADUNLite(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


# 测试代码
if __name__ == "__main__":
    print("测试 DFA-DUN 模型...")

    # 创建标准模型
    model = DFADUN(
        in_channels=3,
        num_stages=4,
        kernel_size=15,
        num_freq_bands=8,
        feat_dim=64,
        num_fmd_blocks=6,
        use_simple_fmd=False
    )

    # 创建轻量模型
    model_lite = DFADUNLite(
        in_channels=3,
        num_stages=2,
        kernel_size=11,
        num_freq_bands=4,
        feat_dim=32,
        num_fmd_blocks=3
    )

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)

    print("\n标准模型测试:")
    with torch.no_grad():
        output, params = model(x, return_params=True)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"估计的 K: {params[0]['K'].shape}")
        print(f"估计的 T: {params[0]['T'].shape}")
        print(f"估计的 D: {params[0]['D'].shape}")
        print(f"估计的 rho: {params[0]['rho'].shape}")

    print("\n轻量模型测试:")
    with torch.no_grad():
        output_lite = model_lite(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output_lite.shape}")
        print(f"参数数量: {sum(p.numel() for p in model_lite.parameters()) / 1e6:.2f}M")

    # 测试阶段输出
    print("\n测试阶段输出:")
    with torch.no_grad():
        stage_output = model.get_stage_output(x, stage_idx=1)
        print(f"第1阶段输出形状: {stage_output.shape}")

    # 测试退化估计
    print("\n测试退化估计:")
    with torch.no_grad():
        degradation = model.estimate_degradation(x)
        print(f"估计的退化参数: {list(degradation.keys())}")

    print("\n所有测试通过!")
