# 数据集加载和使用说明

本项目实现了完整的去雨数据集加载和预处理功能，支持 Rain100L 和 Rain100H 数据集。

## 数据集结构

### Rain100L 数据集结构
```
Rain100L/
├── train/
│   ├── rainy/
│   │   ├── rain-001.png
│   │   ├── rain-002.png
│   │   └── ...
│   ├── norain-001.png
│   ├── norain-002.png
│   └── ...
└── test/
    ├── rainy/
    │   ├── rain-001.png
    │   └── ...
    ├── norain-001.png
    └── ...
```

### Rain100H 数据集结构
```
Rain100H/
├── train/
│   ├── rain-001.png
│   ├── rain-002.png
│   ├── norain-001.png
│   ├── norain-002.png
│   └── ...
└── test/
    ├── rain-001.png
    ├── norain-001.png
    └── ...
```

## 快速开始

### 1. 使用主训练脚本

编辑 `train.py` 中的数据路径：

```python
config['data'] = {
    'dataset_type': 'Rain100L',  # 或 'Rain100H'
    'train_dir': './datasets/Rain100L/train',
    'val_dir': './datasets/Rain100L/test',
    'patch_size': 256,
    'num_workers': 4,
}
```

然后运行：
```bash
python train.py
```

### 2. 使用专用训练脚本

```bash
# 训练 Rain100L
python train_rain100l.py rain100l

# 训练 Rain100H
python train_rain100l.py rain100h
```

### 3. 在代码中直接使用

```python
from utils.datasets import create_dataloader

# 创建训练数据加载器
train_loader = create_dataloader(
    dataset_type='Rain100L',
    root_dir='./datasets/Rain100L/train',
    batch_size=8,
    patch_size=256,
    is_train=True,
    num_workers=4,
    max_samples=None  # None 表示使用全部样本
)

# 创建验证数据加载器
val_loader = create_dataloader(
    dataset_type='Rain100L',
    root_dir='./datasets/Rain100L/test',
    batch_size=1,
    patch_size=256,
    is_train=False,
    num_workers=4
)

# 使用数据加载器
for batch in train_loader:
    rainy, gt, task_label, dummy_kernel = batch
    # rainy: [B, 3, H, W] - 有雨图像
    # gt: [B, 3, H, W] - 无雨图像
    # task_label: [B] - 任务标签 (0=去雨)
    # dummy_kernel: [B, 1, 11, 11] - 虚拟模糊核
    print(f"Rainy shape: {rainy.shape}")
    print(f"GT shape: {gt.shape}")
    break
```

## 数据增强

训练模式下，数据集会自动应用以下增强：

1. **随机裁剪**: 裁剪 256×256 的图像块
2. **随机水平翻转**: 50% 概率
3. **随机垂直翻转**: 50% 概率

验证模式下，使用中心裁剪或调整大小到固定尺寸。

## 参数说明

### `create_dataloader` 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_type` | str | 必需 | 数据集类型: 'Rain100L' 或 'Rain100H' |
| `root_dir` | str | 必需 | 数据集根目录路径 |
| `batch_size` | int | 16 | 批次大小 |
| `patch_size` | int | 256 | 训练时的图像块大小 |
| `is_train` | bool | True | 是否为训练模式 |
| `num_workers` | int | 4 | 数据加载的工作进程数 |
| `max_samples` | int | None | 最大样本数 (用于采样) |

### 配置文件中的数据参数

在 `config` 中配置数据参数：

```python
config = {
    'data': {
        'dataset_type': 'Rain100L',
        'train_dir': './datasets/Rain100L/train',
        'val_dir': './datasets/Rain100L/test',
        'patch_size': 256,
        'num_workers': 4,
        'max_train_samples': None,
        'max_val_samples': None,
    },
    'batch_size': 8,
    # ... 其他配置
}
```

## 测试数据集加载

运行测试脚本验证数据集加载是否正常：

```bash
python -m utils.datasets
```

这将尝试加载并打印数据集信息。

## 常见问题

### Q: 如何处理显存不足？

A: 减小 `batch_size` 或 `patch_size`：

```python
train_loader = create_dataloader(
    dataset_type='Rain100L',
    root_dir='./datasets/Rain100L/train',
    batch_size=4,  # 从 8 减小到 4
    patch_size=128,  # 从 256 减小到 128
    is_train=True
)
```

### Q: 如何使用部分数据进行快速实验？

A: 设置 `max_samples` 参数：

```python
train_loader = create_dataloader(
    dataset_type='Rain100L',
    root_dir='./datasets/Rain100L/train',
    batch_size=8,
    max_samples=100  # 只使用 100 张图片
)
```

### Q: Rain100H 数据集训练集较小怎么办？

A: Rain100H 数据集训练集默认采样 100 张图片。如需使用全部数据：

```python
train_loader = create_dataloader(
    dataset_type='Rain100H',
    root_dir='./datasets/Rain100H/train',
    batch_size=8,
    max_samples=None  # 使用全部数据
)
```

## 数据格式说明

数据加载器返回的每个批次包含 4 个元素：

1. **rainy**: `[B, 3, H, W]` - 有雨图像张量，值范围 [0, 1]
2. **gt**: `[B, 3, H, W]` - 无雨图像张量，值范围 [0, 1]
3. **task_label**: `[B]` - 任务标签，去雨任务为 0
4. **dummy_kernel**: `[B, 1, 11, 11]` - 虚拟模糊核（去雨任务不需要，仅为保持接口一致性）

## 性能优化建议

1. **使用多进程加载**: 设置 `num_workers=4` 或更高以加速数据加载
2. **启用 pin_memory**: DataLoader 已默认启用，可加速 CPU 到 GPU 的数据传输
3. **persistent_workers**: 已默认启用（当 num_workers > 0），可减少 worker 重启开销

## 更多信息

- DFA-DUN 模型相关: 查看 `Restoration/` 目录
- 训练器相关: 查看 `Restoration/trainer.py`
- 配置相关: 查看 `Restoration/config.py`
