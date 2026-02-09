"""
DFA-DUN 训练脚本

支持去雨任务 (Rain100L, Rain100H)
"""

import os
import torch
from Restoration import build_dfa_dun
from Restoration.trainer import DFADUNTrainer
from Restoration.config import get_default_config
from utils.datasets import create_dataloader


def setup_data_loaders(config):
    """
    设置训练和验证数据加载器

    Args:
        config: 配置字典

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    data_config = config.get('data', {})

    # 数据集类型
    dataset_type = data_config.get('dataset_type', 'Rain100L')

    # 训练数据路径
    train_dir = data_config.get('train_dir', './data/Rain100L/train')
    val_dir = data_config.get('val_dir', './data/Rain100L/test')

    # 数据加载参数
    batch_size = config.get('batch_size', 8)
    patch_size = data_config.get('patch_size', 256)
    num_workers = data_config.get('num_workers', 4)
    max_train_samples = data_config.get('max_train_samples', None)
    max_val_samples = data_config.get('max_val_samples', None)

    # 创建训练数据加载器
    print(f"创建训练数据加载器: {dataset_type}")
    train_loader = create_dataloader(
        dataset_type=dataset_type,
        root_dir=train_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        is_train=True,
        num_workers=num_workers,
        max_samples=max_train_samples
    )

    # 创建验证数据加载器
    print(f"创建验证数据加载器: {dataset_type}")
    val_loader = create_dataloader(
        dataset_type=dataset_type,
        root_dir=val_dir,
        batch_size=1,  # 验证时通常使用 batch_size=1
        patch_size=patch_size,
        is_train=False,
        num_workers=num_workers,
        max_samples=max_val_samples
    )

    return train_loader, val_loader


def main():
    """
    主训练函数
    """
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 获取配置
    config = get_default_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'standard')

    print(f"创建模型: {model_type}")
    model = build_dfa_dun(
        model_type=model_type,
        in_channels=model_config.get('in_channels', 3),
        num_stages=model_config.get('num_stages', 4),
        kernel_size=model_config.get('kernel_size', 15),
        num_freq_bands=model_config.get('num_freq_bands', 8),
        feat_dim=model_config.get('feat_dim', 64),
        num_fmd_blocks=model_config.get('num_fmd_blocks', 6)
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建数据加载器
    train_loader, val_loader = setup_data_loaders(config)

    # 创建保存目录
    save_dir = config.get('save_dir', './checkpoints/dfa_dun')
    log_dir = config.get('log_dir', './logs/dfa_dun')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 创建训练器
    trainer = DFADUNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 开始训练
    print("开始训练...")
    trainer.train()


if __name__ == "__main__":
    main()