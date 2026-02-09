"""
数据集测试脚本

用于验证数据集加载和预处理是否正常工作
"""

import os
import sys
import torch
from utils.datasets import create_dataloader


def test_dataloader(dataset_type, root_dir, is_train=True):
    """
    测试数据加载器

    Args:
        dataset_type: 数据集类型 ('Rain100L' 或 'Rain100H')
        root_dir: 数据集根目录
        is_train: 是否为训练模式
    """
    print(f"\n{'='*60}")
    print(f"测试 {dataset_type} 数据集 ({'训练' if is_train else '验证'}模式)")
    print(f"{'='*60}")
    print(f"数据路径: {root_dir}")

    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return False

    try:
        # 创建数据加载器
        dataloader = create_dataloader(
            dataset_type=dataset_type,
            root_dir=root_dir,
            batch_size=2,  # 测试时使用小批次
            patch_size=256,
            is_train=is_train,
            num_workers=0,  # 测试时使用单进程
            max_samples=10  # 测试时只加载少量样本
        )

        print(f"✅ 数据加载器创建成功")
        print(f"   数据集大小: {len(dataloader.dataset)} 张图片")

        # 加载一个批次
        print(f"\n加载第一个批次...")
        for batch_idx, batch in enumerate(dataloader):
            rainy, gt, task_label, dummy_kernel = batch

            print(f"✅ 批次 {batch_idx + 1} 加载成功:")
            print(f"   - Rainy shape: {rainy.shape}")
            print(f"   - GT shape: {gt.shape}")
            print(f"   - Task label: {task_label}")
            print(f"   - Dummy kernel shape: {dummy_kernel.shape}")
            print(f"   - Rainy value range: [{rainy.min():.3f}, {rainy.max():.3f}]")
            print(f"   - GT value range: [{gt.min():.3f}, {gt.max():.3f}]")

            # 只测试第一个批次
            break

        print(f"\n✅ {dataset_type} 数据集测试通过!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("DFA-DUN 数据集测试工具")
    print("="*60)

    # 默认测试路径 (可根据实际情况修改)
    test_paths = {
        'Rain100L_train': './datasets/Rain100L/train',
        'Rain100L_test': './datasets/Rain100L/test',
        'Rain100H_train': './datasets/Rain100H/train',
        'Rain100H_test': './datasets/Rain100H/test',
    }

    # 如果命令行提供了路径，使用命令行路径
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
        print(f"\n使用自定义路径: {custom_path}")

        # 自动检测数据集类型
        if 'Rain100L' in custom_path or 'rain100l' in custom_path.lower():
            if 'train' in custom_path.lower():
                test_dataloader('Rain100L', custom_path, is_train=True)
            else:
                test_dataloader('Rain100L', custom_path, is_train=False)
        elif 'Rain100H' in custom_path or 'rain100h' in custom_path.lower():
            if 'train' in custom_path.lower():
                test_dataloader('Rain100H', custom_path, is_train=True)
            else:
                test_dataloader('Rain100H', custom_path, is_train=False)
        else:
            print(f"⚠️  无法从路径推断数据集类型，尝试两种类型...")

            # 尝试 Rain100L
            if not test_dataloader('Rain100L', custom_path, is_train=True):
                # 如果失败，尝试 Rain100H
                test_dataloader('Rain100H', custom_path, is_train=True)
    else:
        # 使用默认路径进行测试
        print(f"\n使用默认测试路径:")
        for name, path in test_paths.items():
            print(f"  - {name}: {path}")

        print(f"\n提示: 可以通过命令行指定自定义路径")
        print(f"例如: python test_dataset.py /path/to/dataset")

        # 测试所有默认路径
        results = []
        for name, path in test_paths.items():
            if 'train' in name:
                dataset_type = 'Rain100L' if 'Rain100L' in name else 'Rain100H'
                result = test_dataloader(dataset_type, path, is_train=True)
                results.append((name, result))
            else:
                dataset_type = 'Rain100L' if 'Rain100L' in name else 'Rain100H'
                result = test_dataloader(dataset_type, path, is_train=False)
                results.append((name, result))

        # 打印总结
        print(f"\n{'='*60}")
        print("测试总结:")
        print(f"{'='*60}")

        for name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {name}: {status}")

        passed = sum(1 for _, r in results if r)
        total = len(results)
        print(f"\n总计: {passed}/{total} 个测试通过")

        if passed == total:
            print(f"\n🎉 所有测试通过!")
        else:
            print(f"\n⚠️  部分测试失败，请检查数据集路径和格式")


if __name__ == "__main__":
    main()
