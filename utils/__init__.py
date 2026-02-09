"""
工具模块

包含数据集加载、日志记录、指标追踪等工具
"""

import os
from datetime import datetime
from pathlib import Path

from .datasets import (
    BaseRainDataset,
    Rain100LDataset,
    Rain100HDataset,
    create_dataloader,
)

from .logger import (
    TrainingLogger,
    AverageMeter,
    MetricsTracker,
)


def setup_exp_directory(exp_name, timestamp=None, results_root='./results'):
    """
    设置实验目录结构

    创建以下目录结构:
    results/
        exp_name_timestamp/
            checkpoints/  # 模型检查点
            log/          # 训练日志

    Args:
        exp_name: 实验名称
        timestamp: 时间戳字符串,如果为None则使用当前时间
        results_root: 结果根目录

    Returns:
        exp_dir: 实验目录路径
        checkpoints_dir: 检查点目录路径
        log_dir: 日志目录路径
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    exp_dir_name = f"{exp_name}_{timestamp}"
    exp_dir = Path(results_root) / exp_dir_name
    checkpoints_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'log'

    # 创建目录
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return str(exp_dir), str(checkpoints_dir), str(log_dir)


__all__ = [
    'BaseRainDataset',
    'Rain100LDataset',
    'Rain100HDataset',
    'create_dataloader',
    'TrainingLogger',
    'AverageMeter',
    'MetricsTracker',
    'setup_exp_directory',
]
