from Restoration.dfa_dun_v2 import DFADUN, DFADUNLite
import torch

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
