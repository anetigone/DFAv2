"""
DFA-DUN 使用示例

包含:
1. 模型构建
2. 训练流程
3. 推理示例
4. 可视化示例
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.Restoration import (
    DFADUN, DFADUNLite, build_dfa_dun
)
from models.Restoration.trainer import DFADUNTrainer
from models.Restoration.losses import DFADUNLoss
from models.Restoration.config import get_default_config, get_lite_config


# ==========================================
# 示例 1: 构建模型
# ==========================================
def example_build_model():
    """示例: 如何构建 DFA-DUN 模型"""
    print("=" * 60)
    print("示例 1: 构建 DFA-DUN 模型")
    print("=" * 60)

    # 方法 1: 直接构建标准模型
    model = DFADUN(
        in_channels=3,
        num_stages=4,
        kernel_size=15,
        num_freq_bands=8,
        feat_dim=64,
        num_fmd_blocks=6
    )
    print(f"标准模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 方法 2: 构建轻量模型
    model_lite = DFADUNLite(
        in_channels=3,
        num_stages=2,
        kernel_size=11,
        num_freq_bands=4,
        feat_dim=32,
        num_fmd_blocks=3
    )
    print(f"轻量模型参数量: {sum(p.numel() for p in model_lite.parameters()) / 1e6:.2f}M")

    # 方法 3: 使用工厂函数
    model = build_dfa_dun(
        model_type='standard',
        in_channels=3,
        num_stages=4,
        feat_dim=64
    )
    print(f"工厂函数构建模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


# ==========================================
# 示例 2: 前向传播
# ==========================================
def example_forward():
    """示例: 模型前向传播"""
    print("\n" + "=" * 60)
    print("示例 2: 模型前向传播")
    print("=" * 60)

    # 创建模型
    model = DFADUN(
        in_channels=3,
        num_stages=4,
        feat_dim=64
    )
    model.eval()

    # 创建随机输入
    x = torch.randn(2, 3, 256, 256)

    # 前向传播
    with torch.no_grad():
        # 不返回参数
        output = model(x)
        print(f"输出形状: {output.shape}")

        # 返回中间参数 (用于可视化)
        output, params = model(x, return_params=True)
        print(f"输出形状: {output.shape}")
        print(f"参数列表长度: {len(params)}")
        print(f"第一阶段参数:")
        print(f"  - K (模糊核): {params[0]['K'].shape}")
        print(f"  - T (增益): {params[0]['T'].shape}")
        print(f"  - D (偏置): {params[0]['D'].shape}")
        print(f"  - rho (超参数): {params[0]['rho'].shape}")
        print(f"  - freq_mask (频率掩码): {params[0]['freq_mask'].shape}")


# ==========================================
# 示例 3: 损失函数
# ==========================================
def example_loss():
    """示例: 使用损失函数"""
    print("\n" + "=" * 60)
    print("示例 3: 损失函数")
    print("=" * 60)

    # 创建模型
    model = DFADUN(in_channels=3, num_stages=4, feat_dim=64)

    # 创建损失函数
    criterion = DFADUNLoss(
        lambda_rec=1.0,
        lambda_freq=0.1,
        lambda_percep=0.0,
        use_perceptual=False
    )

    # 随机数据
    pred_img = torch.rand(2, 3, 256, 256)
    gt_img = torch.rand(2, 3, 256, 256)

    # 计算损失
    total_loss, loss_dict = criterion(pred_img, gt_img)

    print(f"总损失: {total_loss.item():.4f}")
    print(f"  - 重建损失: {loss_dict['loss_rec'].item():.4f}")
    print(f"  - 频域损失: {loss_dict['loss_freq'].item():.4f}")


# ==========================================
# 示例 4: 训练流程
# ==========================================
def example_training():
    """示例: 训练流程 (伪代码)"""
    print("\n" + "=" * 60)
    print("示例 4: 训练流程")
    print("=" * 60)

    # 1. 获取配置
    config = get_default_config()

    # 2. 构建模型
    model = build_dfa_dun(
        model_type=config['model']['type'],
        **config['model']
    )

    # 3. 创建数据加载器 (需要根据你的数据集实现)
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)

    # 为了示例，创建假的加载器
    class FakeDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.rand(3, 256, 256), torch.rand(3, 256, 256)

    train_dataset = FakeDataset()
    val_dataset = FakeDataset()

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 4. 创建训练器
    trainer = DFADUNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 5. 开始训练
    print("开始训练...")
    # trainer.train()  # 取消注释以实际训练

    print("训练完成! (这是示例，未实际运行)")


# ==========================================
# 示例 5: 推理
# ==========================================
def example_inference():
    """示例: 推理流程"""
    print("\n" + "=" * 60)
    print("示例 5: 推理流程")
    print("=" * 60)

    # 1. 加载模型
    model = DFADUN(in_channels=3, num_stages=4)

    # 2. 加载权重 (如果有保存的权重)
    # checkpoint = torch.load('checkpoints/best_model.pth')
    # model.load_state_dict(checkpoint['state_dict'])
    print("模型已加载 (未加载实际权重)")

    # 3. 设置为评估模式
    model.eval()

    # 4. 推理
    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")


# ==========================================
# 示例 6: 可视化
# ==========================================
def example_visualization():
    """示例: 可视化退化参数"""
    print("\n" + "=" * 60)
    print("示例 6: 可视化")
    print("=" * 60)

    # 创建模型
    model = DFADUN(in_channels=3, num_stages=4)
    model.eval()

    # 创建输入
    x = torch.randn(1, 3, 256, 256)

    # 获取参数
    with torch.no_grad():
        output, params = model(x, return_params=True)

    # 可视化第一阶段估计的参数
    stage_idx = 0
    K = params[stage_idx]['K'][0, 0].cpu().numpy()  # [kH, kW]
    T = params[stage_idx]['T'][0, 0].cpu().numpy()  # [H, W]
    D = params[stage_idx]['D'][0, 0].cpu().numpy()  # [H, W]
    freq_mask = params[stage_idx]['freq_mask'][0, 0].cpu().numpy()  # [H, W//2+1]

    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 模糊核
    axes[0, 0].imshow(K, cmap='hot')
    axes[0, 0].set_title('Estimated Blur Kernel (K)')
    axes[0, 0].axis('off')

    # 增益图
    im1 = axes[0, 1].imshow(T, cmap='jet')
    axes[0, 1].set_title('Estimated Gain Map (T)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])

    # 偏置图
    im2 = axes[1, 0].imshow(D, cmap='coolwarm')
    axes[1, 0].set_title('Estimated Bias Map (D)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])

    # 频率掩码
    im3 = axes[1, 1].imshow(freq_mask, cmap='viridis')
    axes[1, 1].set_title('Frequency Mask')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('dfa_dun_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 dfa_dun_visualization.png")


# ==========================================
# 示例 7: 估计退化类型
# ==========================================
def example_estimate_degradation():
    """示例: 估计退化类型"""
    print("\n" + "=" * 60)
    print("示例 7: 估计退化类型")
    print("=" * 60)

    # 创建模型
    model = DFADUN(in_channels=3, num_stages=4)

    # 创建不同类型的退化图像 (模拟)
    # 模糊图像
    blur_img = torch.randn(1, 3, 256, 256) * 0.5 + 0.5

    # 噪声图像
    noise_img = torch.randn(1, 3, 256, 256) * 0.2 + 0.3

    # 低光照图像
    low_light_img = torch.randn(1, 3, 256, 256) * 0.3 + 0.1

    images = {
        'Blurred': blur_img,
        'Noisy': noise_img,
        'Low Light': low_light_img
    }

    # 估计退化参数
    model.eval()
    with torch.no_grad():
        for name, img in images.items():
            params = model.estimate_degradation(img)

            print(f"\n{name} Image:")
            print(f"  - 平均增益 T: {params['T'].mean().item():.4f}")
            print(f"  - 平均偏置 D: {params['D'].mean().item():.4f}")
            print(f"  - 超参数 rho: {params['rho'].mean().item():.4f}")

            # 根据参数判断退化类型
            T_mean = params['T'].mean().item()
            D_mean = params['D'].mean().item()

            if T_mean < 0.7:
                pred_type = "Low Light / Underexposed"
            elif D_mean > 0.1 or D_mean < -0.1:
                pred_type = "Additive Degradation (Rain/Snow)"
            else:
                pred_type = "Blur / Convolutional Degradation"

            print(f"  - 预测类型: {pred_type}")


# ==========================================
# 主函数
# ==========================================
def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DFA-DUN 使用示例")
    print("=" * 60)

    # 运行示例
    example_build_model()
    example_forward()
    example_loss()
    example_training()
    example_inference()

    # 可视化示例需要 matplotlib
    try:
        example_visualization()
    except Exception as e:
        print(f"\n可视化跳过 (需要 matplotlib): {e}")

    example_estimate_degradation()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
