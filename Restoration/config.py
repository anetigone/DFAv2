"""
DFA-DUN 配置文件

包含模型、训练和数据的默认配置
"""

import torch


def get_default_config():
    """
    获取默认配置

    Returns:
        config: 配置字典
    """
    config = {
        # ========== 模型配置 ==========
        'model': {
            'type': 'standard',  # 'standard' 或 'lite'
            'in_channels': 3,
            'num_stages': 4,
            'kernel_size': 15,
            'num_freq_bands': 8,
            'feat_dim': 64,
            'num_fmd_blocks': 6,
            'use_simple_fmd': False
        },

        # ========== 训练配置 ==========
        'epochs': 100,
        'batch_size': 8,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adam',  # 'adam' 或 'adamw'
        'scheduler': 'cosine',  # 'cosine', 'step', 'plateau'
        'grad_clip': 1.0,
        'use_amp': True,  # 混合精度训练

        # ========== 损失函数配置 ==========
        'lambda_rec': 1.0,  # 重建损失权重
        'lambda_freq': 0.1,  # 频域损失权重
        'lambda_percep': 0.0,  # 感知损失权重
        'lambda_param': 0.0,  # 参数监督损失权重
        'use_perceptual': False,  # 是否使用感知损失
        'use_parameter': False,  # 是否使用参数监督损失

        # ========== 数据配置 ==========
        'data': {
            'train_dir': './data/train',
            'val_dir': './data/val',
            'num_workers': 4,
            'patch_size': 256,
            'use_augmentation': True
        },

        # ========== 保存和日志配置 ==========
        'save_dir': './checkpoints/dfa_dun',
        'log_dir': './logs/dfa_dun',
        'exp_name': 'dfa_dun_all_in_one',
        'save_interval': 10,
        'val_interval': 1,
        'log_interval': 10,

        # ========== 其他配置 ==========
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }

    return config


def get_lite_config():
    """
    获取轻量版配置 (用于快速实验)

    Returns:
        config: 配置字典
    """
    config = get_default_config()

    # 修改模型配置
    config['model']['type'] = 'lite'
    config['model']['num_stages'] = 2
    config['model']['kernel_size'] = 11
    config['model']['num_freq_bands'] = 4
    config['model']['feat_dim'] = 32
    config['model']['num_fmd_blocks'] = 3

    # 修改训练配置
    config['batch_size'] = 16
    config['epochs'] = 50

    return config


def get_large_config():
    """
    获取大模型配置 (用于最佳性能)

    Returns:
        config: 配置字典
    """
    config = get_default_config()

    # 修改模型配置
    config['model']['num_stages'] = 6
    config['model']['kernel_size'] = 21
    config['model']['num_freq_bands'] = 12
    config['model']['feat_dim'] = 128
    config['model']['num_fmd_blocks'] = 10

    # 修改训练配置
    config['batch_size'] = 4
    config['epochs'] = 200
    config['lr'] = 5e-5

    # 启用感知损失
    config['lambda_percep'] = 0.05
    config['use_perceptual'] = True

    return config


# 配置验证
def validate_config(config):
    """
    验证配置的合法性

    Args:
        config: 配置字典

    Returns:
        is_valid: 是否合法
        errors: 错误信息列表
    """
    errors = []

    # 检查必需的键
    required_keys = [
        'model', 'epochs', 'batch_size', 'lr',
        'save_dir', 'log_dir', 'exp_name'
    ]

    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    # 检查模型配置
    if 'model' in config:
        model_keys = ['type', 'in_channels', 'num_stages']
        for key in model_keys:
            if key not in config['model']:
                errors.append(f"Missing required model key: {key}")

        # 检查模型类型
        if config['model']['type'] not in ['standard', 'lite']:
            errors.append(f"Invalid model type: {config['model']['type']}")

    # 检查数值范围
    if config.get('batch_size', 0) <= 0:
        errors.append("batch_size must be positive")

    if config.get('lr', 0) <= 0:
        errors.append("lr must be positive")

    if config.get('epochs', 0) <= 0:
        errors.append("epochs must be positive")

    is_valid = len(errors) == 0

    return is_valid, errors


# 打印配置
def print_config(config):
    """
    打印配置信息

    Args:
        config: 配置字典
    """
    print("=" * 50)
    print("DFA-DUN Configuration")
    print("=" * 50)

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_dict(config)

    print("=" * 50)


if __name__ == "__main__":
    # 测试配置
    print("默认配置:")
    config = get_default_config()
    print_config(config)

    print("\n验证配置:")
    is_valid, errors = validate_config(config)
    if is_valid:
        print("✓ 配置合法")
    else:
        print("✗ 配置错误:")
        for error in errors:
            print(f"  - {error}")

    print("\n轻量版配置:")
    lite_config = get_lite_config()
    print_config(lite_config)

    print("\n大模型配置:")
    large_config = get_large_config()
    print_config(large_config)
