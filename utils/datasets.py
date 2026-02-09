"""
去雨数据集加载模块

支持:
1. Rain100L 数据集
2. Rain100H 数据集
3. 数据增强和预处理
4. 训练和验证模式
"""

import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


class BaseRainDataset(Dataset):
    """去雨数据集基类，支持 norain-xxx 和 rain-xxx 命名格式"""

    def __init__(self, root_dir, patch_size=256, is_train=True, max_samples=None):
        """
        Args:
            root_dir: 数据集根目录
            patch_size: 训练时裁剪的大小
            is_train: 是否为训练模式
            max_samples: 最大样本数，用于采样。None表示使用全部样本
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.is_train = is_train

        # 设置随机种子以确保可重复性
        random.seed(42)

        # 获取所有图片对
        self.rainy_images, self.gt_images = self._load_image_pairs()

        # 采样逻辑
        if max_samples is not None and len(self.rainy_images) > max_samples:
            indices = random.sample(range(len(self.rainy_images)), max_samples)
            self.rainy_images = [self.rainy_images[i] for i in indices]
            self.gt_images = [self.gt_images[i] for i in indices]
            print(f"从 {len(self.rainy_images)} 张图片中采样了 {max_samples} 张")

        print(f"加载了 {len(self.rainy_images)} 张图片对")

    def _load_image_pairs(self):
        """加载图片对，子类可以重写此方法以支持不同的目录结构"""
        # 默认实现：同一目录下的 rain-xxx.png 和 norain-xxx.png
        rainy_images = sorted(glob.glob(os.path.join(self.root_dir, 'rain-*.png')))

        gt_images = []
        for r_path in rainy_images:
            # 提取编号，例如从 rain-001.png 提取 001
            idx = os.path.basename(r_path).split('-')[-1]
            gt_path = os.path.join(self.root_dir, f'norain-{idx}')
            gt_images.append(gt_path)

        return rainy_images, gt_images

    def __len__(self):
        return len(self.rainy_images)

    def transform(self, rainy, gt):
        """
        数据增强和预处理

        Args:
            rainy: 有雨图像
            gt: 无雨图像

        Returns:
            rainy_tensor, gt_tensor: 处理后的张量
        """
        # 训练模式：随机裁剪和翻转
        if self.is_train:
            # 随机裁剪
            width, height = rainy.size

            # 确保图像足够大
            if width < self.patch_size or height < self.patch_size:
                # 如果图像太小，先调整大小
                scale = max(self.patch_size / width, self.patch_size / height)
                new_width = int(width * scale) + 1
                new_height = int(height * scale) + 1
                rainy = rainy.resize((new_width, new_height), Image.BICUBIC)
                gt = gt.resize((new_width, new_height), Image.BICUBIC)
                width, height = new_width, new_height

            # 随机裁剪
            i = random.randint(0, height - self.patch_size)
            j = random.randint(0, width - self.patch_size)
            h = w = self.patch_size
            rainy = F.crop(rainy, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)

            # 随机水平翻转
            if random.random() > 0.5:
                rainy = F.hflip(rainy)
                gt = F.hflip(gt)

            # 随机垂直翻转
            if random.random() > 0.5:
                rainy = F.vflip(rainy)
                gt = F.vflip(gt)
        else:
            # 测试模式：中心裁剪或调整大小
            width, height = rainy.size
            if width >= self.patch_size and height >= self.patch_size:
                rainy = F.center_crop(rainy, (self.patch_size, self.patch_size))
                gt = F.center_crop(gt, (self.patch_size, self.patch_size))
            else:
                # 如果图像太小，调整大小到 patch_size
                rainy = rainy.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                gt = gt.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # 转为 Tensor
        rainy = F.to_tensor(rainy)
        gt = F.to_tensor(gt)

        return rainy, gt

    def __getitem__(self, index):
        """
        获取一个样本

        Returns:
            rainy_tensor: 有雨图像张量
            gt_tensor: 无雨图像张量
            task_label: 任务标签 (0=去雨)
            dummy_kernel: 虚拟模糊核 (去雨任务不需要)
        """
        rainy_img = Image.open(self.rainy_images[index]).convert('RGB')
        gt_img = Image.open(self.gt_images[index]).convert('RGB')

        rainy_tensor, gt_tensor = self.transform(rainy_img, gt_img)

        # 针对 Rain100L 任务的特定标签：
        # 根据之前的设计，0 代表 Identity (去雨/去噪)，1 代表 Convolution (去模糊)
        task_label = torch.tensor(0).long()

        # 因为去雨不需要模糊核，提供一个虚拟的单位核 (1x1) 或全 0 占位
        dummy_kernel = torch.zeros((1, 11, 11))

        return rainy_tensor, gt_tensor, task_label, dummy_kernel


class Rain100LDataset(BaseRainDataset):
    """Rain100L数据集：图片位于 rainy 子目录中"""

    def __init__(self, root_dir, patch_size=256, is_train=True, max_samples=None):
        """
        Args:
            root_dir: Rain100L 文件夹路径
            patch_size: 训练时裁剪的大小
            is_train: 是否为训练模式
            max_samples: 最大样本数，None表示使用全部样本
        """
        # 保存原始root_dir，父类会使用它
        self.original_root_dir = root_dir
        # 调用父类__init__，但会在 _load_image_pairs 中处理子目录
        super().__init__(root_dir, patch_size, is_train, max_samples)

    def _load_image_pairs(self):
        """Rain100L的图片在rainy子目录下"""
        rainy_images = sorted(glob.glob(os.path.join(self.original_root_dir, 'rainy', 'rain-*.png')))

        gt_images = []
        for r_path in rainy_images:
            # 提取编号，例如从 rain-001.png 提取 001
            idx = os.path.basename(r_path).split('-')[-1]
            # Rain100L的ground truth在同一目录下
            gt_path = os.path.join(self.original_root_dir, f'norain-{idx}')
            gt_images.append(gt_path)

        return rainy_images, gt_images


class Rain100HDataset(BaseRainDataset):
    """Rain100H数据集：图片位于同一目录，支持采样"""

    def __init__(self, root_dir, patch_size=256, is_train=True, max_samples=100):
        """
        Args:
            root_dir: Rain100H 文件夹路径 (rain-xxx.png 和 norain-xxx.png 在同一目录)
            patch_size: 训练时裁剪的大小
            is_train: 是否为训练模式
            max_samples: 最大样本数，默认100张
        """
        super().__init__(root_dir, patch_size, is_train, max_samples)

    def _load_image_pairs(self):
        """Rain100H的rain和norain图片在同一目录下"""
        # 默认基类实现已经支持同一目录的格式
        return super()._load_image_pairs()


def create_dataloader(dataset_type, root_dir, batch_size=16, patch_size=256,
                      is_train=True, num_workers=4, max_samples=None):
    """
    创建数据加载器

    Args:
        dataset_type: 数据集类型，可选 'Rain100L' 或 'Rain100H'
        root_dir: 数据集根目录
        batch_size: 批次大小
        patch_size: 训练时裁剪的大小
        is_train: 是否为训练模式
        num_workers: 数据加载的工作进程数
        max_samples: 最大样本数，用于采样。Rain100H默认100张，Rain100L默认全部

    Returns:
        DataLoader对象
    """
    # 根据数据集类型创建相应的dataset
    if dataset_type == 'Rain100L':
        dataset = Rain100LDataset(
            root_dir,
            patch_size=patch_size,
            is_train=is_train,
            max_samples=max_samples
        )
    elif dataset_type == 'Rain100H':
        # Rain100H默认采样100张
        if max_samples is None:
            max_samples = 100
        dataset = Rain100HDataset(
            root_dir,
            patch_size=patch_size,
            is_train=is_train,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}。请选择 'Rain100L' 或 'Rain100H'")

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=num_workers > 0  # 保持 worker 进程以提高效率
    )

    return dataloader


if __name__ == "__main__":
    """
    测试数据集加载
    """
    # 测试 Rain100L 数据集
    print("测试 Rain100L 数据集:")
    rain100l_train = create_dataloader(
        dataset_type='Rain100L',
        root_dir='./data/Rain100L/train',
        batch_size=4,
        patch_size=256,
        is_train=True,
        num_workers=0,
        max_samples=10  # 测试时只加载10张图片
    )
    # 测试加载一个批次
    for batch in rain100l_train:
        rainy, gt, task_label, dummy_kernel = batch
        print(f"  Rainy shape: {rainy.shape}")
        print(f"  GT shape: {gt.shape}")
        print(f"  Task label: {task_label}")
        print(f"  Dummy kernel shape: {dummy_kernel.shape}")
    # 测试 Rain100H 数据集
    print("\n测试 Rain100H 数据集:")
    rain100h_train = create_dataloader(
        dataset_type='Rain100H',
        root_dir='./data/Rain100H/train',
        batch_size=4,
        patch_size=256,
        is_train=True,
        num_workers=0,
        max_samples=10
    )

    # 测试加载一个批次
    for batch in rain100h_train:
        rainy, gt, task_label, dummy_kernel = batch
        print(f"  Rainy shape: {rainy.shape}")
        print(f"  GT shape: {gt.shape}")
        print(f"  Task label: {task_label}")
        print(f"  Dummy kernel shape: {dummy_kernel.shape}")
        break
