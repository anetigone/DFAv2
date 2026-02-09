"""
Rain100L 训练示例

演示如何配置和训练 DFA-DUN 模型用于去雨任务
"""

import os
import torch
from Restoration import build_dfa_dun
from Restoration.trainer import DFADUNTrainer
from Restoration.config import get_default_config
from utils.datasets import create_dataloader


def train_rain100l():
    """
    训练 Rain100L 数据集
    """
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 获取配置
    config = get_default_config()

    # 修改数据配置
    config['data'] = {
        'dataset_type': 'Rain100L',
        'train_dir': './datasets/Rain100L/train',
        'val_dir': './datasets/Rain100L/test',
        'patch_size': 256,
        'num_workers': 4,
        'max_train_samples': None,  # 使用全部训练数据
        'max_val_samples': None,    # 使用全部验证数据
    }

    # 修改训练配置
    config['batch_size'] = 8  # 根据 GPU 显存调整
    config['epochs'] = 100
    config['lr'] = 1e-4

    # 修改保存路径
    config['save_dir'] = './checkpoints/rain100l'
    config['log_dir'] = './logs/rain100l'
    config['exp_name'] = 'dfa_dun_rain100l'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = build_dfa_dun(
        model_type='standard',
        in_channels=3,
        num_stages=4,
        kernel_size=15,
        num_freq_bands=8,
        feat_dim=64,
        num_fmd_blocks=6
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_dataloader(
        dataset_type='Rain100L',
        root_dir=config['data']['train_dir'],
        batch_size=config['batch_size'],
        patch_size=config['data']['patch_size'],
        is_train=True,
        num_workers=config['data']['num_workers'],
        max_samples=config['data']['max_train_samples']
    )

    val_loader = create_dataloader(
        dataset_type='Rain100L',
        root_dir=config['data']['val_dir'],
        batch_size=1,
        patch_size=config['data']['patch_size'],
        is_train=False,
        num_workers=config['data']['num_workers'],
        max_samples=config['data']['max_val_samples']
    )

    # 创建训练器
    trainer = DFADUNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 开始训练
    print("开始训练 Rain100L...")
    trainer.train()


def train_rain100h():
    """
    训练 Rain100H 数据集
    """
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 获取配置
    config = get_default_config()

    # 修改数据配置
    config['data'] = {
        'dataset_type': 'Rain100H',
        'train_dir': './datasets/Rain100H/train',
        'val_dir': './datasets/Rain100H/test',
        'patch_size': 256,
        'num_workers': 4,
        'max_train_samples': 100,  # Rain100H 训练集较小
        'max_val_samples': None,
    }

    # 修改训练配置
    config['batch_size'] = 8
    config['epochs'] = 100
    config['lr'] = 1e-4

    # 修改保存路径
    config['save_dir'] = './checkpoints/rain100h'
    config['log_dir'] = './logs/rain100h'
    config['exp_name'] = 'dfa_dun_rain100h'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = build_dfa_dun(
        model_type='standard',
        in_channels=3,
        num_stages=4,
        kernel_size=15,
        num_freq_bands=8,
        feat_dim=64,
        num_fmd_blocks=6
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_dataloader(
        dataset_type='Rain100H',
        root_dir=config['data']['train_dir'],
        batch_size=config['batch_size'],
        patch_size=config['data']['patch_size'],
        is_train=True,
        num_workers=config['data']['num_workers'],
        max_samples=config['data']['max_train_samples']
    )

    val_loader = create_dataloader(
        dataset_type='Rain100H',
        root_dir=config['data']['val_dir'],
        batch_size=1,
        patch_size=config['data']['patch_size'],
        is_train=False,
        num_workers=config['data']['num_workers'],
        max_samples=config['data']['max_val_samples']
    )

    # 创建训练器
    trainer = DFADUNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 开始训练
    print("开始训练 Rain100H...")
    trainer.train()


if __name__ == "__main__":
    # 选择要训练的数据集
    import sys

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        print("使用方法:")
        print("  python train_rain100l.py rain100l  # 训练 Rain100L")
        print("  python train_rain100l.py rain100h  # 训练 Rain100H")
        exit()

    if dataset.lower() == 'rain100l':
        train_rain100l()
    elif dataset.lower() == 'rain100h':
        train_rain100h()
    else:
        print(f"未知的数据集: {dataset}")
        print("请选择 'rain100l' 或 'rain100h'")
