# DFA-DUN: Dynamic Frequency-Aware Deep Unfolding Network

基于 HQS (Half Quadratic Splitting) 的 All-in-One 图像恢复模型，统一处理去雨、去噪、去雾、去模糊、低光照增强等任务。

## 核心思想

### 统一物理模型

DFA-DUN 将多种图像退化过程统一为一个物理公式:

```
y = K ⊗ (x ⊙ T) + D
```

其中:
- `y`: 观测到的退化图像
- `x`: 需要恢复的清晰图像
- `K`: 模糊核 (卷积退化)
- `T`: 增益图 (像素级亮度调整)
- `D`: 偏置图 (加性退化，如雨痕、雾)
- `⊗`: 卷积操作
- `⊙`: 逐元素乘法

### 优化问题

将图像恢复转化为以下优化问题:

```
min_x 1/2 ||y - K ⊗ (x ⊙ T) - D||^2 + λΦ(x)
```

通过 HQS 算法展开，每个迭代阶段包含:

1. **DFAP**: 动态频率感知参数估计器
2. **DFU**: 数据保真单元 (HQS解析解)
3. **FMD**: 频率调制恢复器 (深度先验)

## 架构设计

```
Input y
    ↓
[Stage 1]
    ├─→ DFAP: 估计 K, T, D, rho, freq_mask
    ├─→ DFU: HQS 解析解
    └─→ FMD: 频率调制深度恢复
    ↓
[Stage 2]
    ├─→ DFAP
    ├─→ DFU
    └─→ FMD
    ↓
...
    ↓
[Stage S]
    ↓
Output x
```

## 文件结构

```
models/Restoration/
├── __init__.py           # 模块导出
├── layers.py             # 基础层 (LayerNorm, 注意力机制)
├── dfap.py              # 动态频率感知参数估计器
├── dfu.py               # 数据保真单元
├── fmd.py               # 频率调制恢复器
├── dfa_dun_v2.py        # DFA-DUN 主模型
├── losses.py            # 损失函数
├── trainer.py           # 训练器
├── config.py            # 配置文件
├── example.py           # 使用示例
└── README.md            # 本文件
```

## 快速开始

### 1. 构建模型

```python
from models.Restoration import build_dfa_dun

# 标准模型
model = build_dfa_dun(
    model_type='standard',
    in_channels=3,
    num_stages=4,
    kernel_size=15,
    num_freq_bands=8,
    feat_dim=64,
    num_fmd_blocks=6
)

# 轻量模型
model_lite = build_dfa_dun(
    model_type='lite',
    in_channels=3,
    num_stages=2,
    feat_dim=32
)
```

### 2. 训练

```python
from models.Restoration.trainer import DFADUNTrainer
from models.Restoration.config import get_default_config

# 获取配置
config = get_default_config()

# 创建训练器
trainer = DFADUNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# 开始训练
trainer.train()
```

### 3. 推理

```python
import torch

# 加载模型
model.eval()

# 推理
with torch.no_grad():
    output = model(degraded_image)
```

### 4. 估计退化参数

```python
# 仅估计退化类型 (不恢复)
params = model.estimate_degradation(degraded_image)

print(f"估计的模糊核: {params['K'].shape}")
print(f"估计的增益图: {params['T'].shape}")
print(f"估计的偏置图: {params['D'].shape}")
print(f"频率掩码: {params['freq_mask'].shape}")
```

## 配置选项

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_stages` | 展开阶段数 | 4 |
| `kernel_size` | 预测模糊核大小 | 15 |
| `num_freq_bands` | 频率分析带数 | 8 |
| `feat_dim` | 特征维度 | 64 |
| `num_fmd_blocks` | FMD 模块数 | 6 |

### 训练配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `epochs` | 训练轮数 | 100 |
| `batch_size` | 批大小 | 8 |
| `lr` | 学习率 | 1e-4 |
| `optimizer` | 优化器 | adam |
| `scheduler` | 学习率调度器 | cosine |
| `use_amp` | 混合精度训练 | True |

### 损失函数配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `lambda_rec` | 重建损失权重 | 1.0 |
| `lambda_freq` | 频域损失权重 | 0.1 |
| `lambda_percep` | 感知损失权重 | 0.0 |
| `lambda_param` | 参数监督损失权重 | 0.0 |

## 核心模块

### DFAP (Dynamic Frequency-Aware Parameter Estimator)

动态估计退化参数:
- **K**: 模糊核
- **T**: 增益图
- **D**: 偏置图
- **rho**: 超参数
- **freq_mask**: 频率掩码

### DFU (Data Fidelity Unit)

实现 HQS 解析解:

```python
z_k = [T ⊙ (K^T ⊗ (y - D)) + rho * x_{k-1}] / [T^2 ⊙ (K^T ⊗ K) + rho]
```

### FMD (Frequency-Modulated Denoiser)

深度先验网络，包含:
- 频率门控注意力 (FGA)
- 自适应频率调制 (AdaFM)
- CBAM 注意力机制

## 支持的任务

- **去雨**: 估计 D (雨痕层) 并减去
- **去噪**: K ≈ 1, T ≈ 1，主要依赖深度先验
- **去雾**: 估计 T (透射率) 和 D (大气光)
- **去模糊**: 估计 K (模糊核) 并去卷积
- **低光照增强**: 估计 T (增益图) 提升亮度

## 实验结果

(待补充)

## 引用

如果这个项目对您有帮助，请考虑引用:

```bibtex
@article{dfa_dun,
  title={DFA-DUN: Dynamic Frequency-Aware Deep Unfolding Network for All-in-One Image Restoration},
  author={...},
  journal={...},
  year={2025}
}
```

## 许可证

MIT License

## 联系方式

如有问题，请联系项目维护者。
