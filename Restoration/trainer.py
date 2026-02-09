"""
DFA-DUN 训练器

支持:
1. All-in-One 训练 (混合多种退化)
2. 单任务训练
3. 验证和模型保存
4. 日志记录
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from .losses import DFADUNLoss
from utils import TrainingLogger, MetricsTracker


class DFADUNTrainer:
    """
    DFA-DUN 训练器

    Args:
        model: DFA-DUN 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置字典
        device: 训练设备
        logger: 日志记录器 (可选)
    """
    def __init__(self, model, train_loader, val_loader, config, device=None, logger=None):
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 模型
        self.model = model.to(self.device)

        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 配置
        self.config = config
        self.num_epochs = config.get('epochs', 100)

        # 创建实验目录结构: results/exp_name_timestamp/checkpoints 和 results/exp_name_timestamp/log
        exp_name = config.get('exp_name', 'dfa_dun')
        timestamp = config.get('timestamp', '')

        if not timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.exp_dir_name = f"{exp_name}_{timestamp}"
        self.exp_dir = os.path.join('./results', self.exp_dir_name)

        self.save_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'log')

        # 创建目录结构
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 损失函数
        self.criterion = DFADUNLoss(
            lambda_rec=config.get('lambda_rec', 1.0),
            lambda_freq=config.get('lambda_freq', 0.1),
            lambda_percep=config.get('lambda_percep', 0.0),
            lambda_param=config.get('lambda_param', 0.0),
            use_perceptual=config.get('use_perceptual', False),
            use_parameter=config.get('use_parameter', False)
        )

        # 优化器
        optimizer_type = config.get('optimizer', 'adam')
        lr = config.get('lr', 1e-4)
        weight_decay = config.get('weight_decay', 0)

        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # 学习率调度器
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.5)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None

        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()

        # 最佳指标
        self.best_psnr = 0.0
        self.best_loss = float('inf')

        # 日志记录器
        if logger is None:
            self.logger = TrainingLogger(
                log_dir=self.log_dir,
                exp_name=None,  # 不再创建额外的子目录
                create_exp_subdir=False
            )
        else:
            self.logger = logger

        # 记录配置
        self.logger.log_config(config)

    def train_epoch(self, epoch):
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        metrics_tracker = MetricsTracker()

        # 当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_epoch_start(epoch, self.num_epochs, current_lr)

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 数据解包 (根据你的数据集格式调整)
            # 假设格式: (degraded, gt)
            if len(batch) == 2:
                img_degraded, img_gt = batch
                img_degraded = img_degraded.to(self.device)
                img_gt = img_gt.to(self.device)
            elif len(batch) >= 3:
                # 如果有额外信息 (如退化类型、模糊核等)
                img_degraded, img_gt, *_ = batch
                img_degraded = img_degraded.to(self.device)
                img_gt = img_gt.to(self.device)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)}")

            self.optimizer.zero_grad()

            # 前向传播
            if self.use_amp:
                with autocast("cuda"):
                    # 模型输出
                    output, params = self.model(img_degraded)

                    # 计算损失
                    loss, loss_dict = self.criterion(output, img_gt, pred_params=params)

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )

                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 不使用混合精度
                output, params = self.model(img_degraded)
                loss, loss_dict = self.criterion(output, img_gt, pred_params=params)

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )

                self.optimizer.step()

            # 检查 NaN
            if torch.isnan(loss):
                self.logger.logger.error(f"NaN loss detected at epoch {epoch}, batch {batch_idx}!")
                continue

            # 更新指标
            metrics = {
                'loss': loss.item(),
                'loss_rec': loss_dict['loss_rec'].item(),
                'loss_freq': loss_dict['loss_freq'].item(),
                'loss_consist': loss_dict['loss_consist'].item()
            }

            if self.criterion.use_perceptual:
                metrics['loss_percep'] = loss_dict['loss_percep'].item()

            if self.criterion.use_parameter:
                metrics['loss_param'] = loss_dict['loss_param'].item()

            metrics_tracker.update(metrics, n=img_degraded.size(0))

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Rec': f"{loss_dict['loss_rec'].item():.4f}",
                'Freq': f"{loss_dict['loss_freq'].item():.4f}",
                'Consist': f"{loss_dict['loss_consist'].item():.4f}"
            })

            # 记录日志
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.log_iter(
                    epoch=epoch,
                    iteration=batch_idx,
                    total_iters=len(self.train_loader),
                    metrics=metrics
                )

        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()

        # 记录 epoch 结束
        epoch_metrics = metrics_tracker.get_metrics()
        self.logger.log_epoch_end(epoch, epoch_metrics, phase='train')

        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch):
        """
        验证

        Args:
            epoch: 当前 epoch 编号

        Returns:
            metrics: 验证指标字典
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        pbar = tqdm(self.val_loader, desc=f"Validation")

        for batch_idx, batch in enumerate(pbar):
            # 数据解包
            if len(batch) == 2:
                img_degraded, img_gt = batch
                img_degraded = img_degraded.to(self.device)
                img_gt = img_gt.to(self.device)
            elif len(batch) >= 3:
                img_degraded, img_gt, *_ = batch
                img_degraded = img_degraded.to(self.device)
                img_gt = img_gt.to(self.device)

            # 前向传播
            output, params = self.model(img_degraded)

            # 计算损失
            loss, loss_dict = self.criterion(output, img_gt, pred_params=params)

            # 计算 PSNR
            mse = torch.mean((output - img_gt) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

            # 更新指标
            metrics = {
                'loss': loss.item(),
                'psnr': psnr.item()
            }
            metrics_tracker.update(metrics, n=img_degraded.size(0))

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{psnr.item():.2f}"
            })

        # 平均指标
        val_metrics = metrics_tracker.get_metrics()

        # 记录验证结果
        self.logger.log_epoch_end(epoch, val_metrics, phase='val')

        # 保存最佳模型
        if val_metrics['psnr'] > self.best_psnr:
            self.best_psnr = val_metrics['psnr']
            self.save_checkpoint("best_model_psnr.pth")
            self.logger.log_best_metric(epoch, 'PSNR', self.best_psnr)

        if val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            self.save_checkpoint("best_model_loss.pth")

        return val_metrics

    def save_checkpoint(self, filename):
        """
        保存检查点

        Args:
            filename: 保存的文件名
        """
        checkpoint_path = os.path.join(self.save_dir, filename)

        torch.save({
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else None,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_psnr': self.best_psnr,
            'best_loss': self.best_loss,
            'config': self.config
        }, checkpoint_path)

        # 记录相对路径，更简洁
        self.logger.logger.info(f"检查点已保存: {filename}")

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and checkpoint.get('scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if checkpoint.get('epoch') is not None:
            self.current_epoch = checkpoint['epoch']

        # 使用相对路径显示
        checkpoint_name = os.path.basename(checkpoint_path)
        self.logger.logger.info(f"检查点已加载: {checkpoint_name}")
        self.logger.logger.info(f"最佳 PSNR: {self.best_psnr:.4f}")

    def train(self):
        """
        完整训练流程
        """
        self.logger.logger.info("开始训练...")

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            if epoch % self.config.get('val_interval', 1) == 0:
                val_metrics = self.validate(epoch)

            # 保存定期检查点
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

        self.logger.logger.info("训练完成!")
        self.logger.logger.info(f"最佳 PSNR: {self.best_psnr:.4f}")
