"""
DFA-DUN 损失函数

包含:
1. 重建损失 (L1 / Charbonnier)
2. 频域损失 (FFT Loss)
3. 感知损失 (Perceptual Loss)
4. 物理参数监督损失 (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 的平滑版本)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x: 预测图像 [B, C, H, W]
            y: 真值图像 [B, C, H, W]
        Returns:
            loss: 标量
        """
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class FrequencyLoss(nn.Module):
    """
    频域损失 (Frequency Domain Loss)

    在频域计算 L1 距离，帮助模型更好地恢复频谱信息
    """
    def __init__(self, use_amplitude=True, use_phase=False):
        super(FrequencyLoss, self).__init__()
        self.use_amplitude = use_amplitude
        self.use_phase = use_phase

    def forward(self, x, y):
        """
        Args:
            x: 预测图像 [B, C, H, W]
            y: 真值图像 [B, C, H, W]
        Returns:
            loss: 标量
        """
        # 转换到频域
        x_fft = torch.fft.rfft2(x, norm='ortho')
        y_fft = torch.fft.rfft2(y, norm='ortho')

        loss = 0.0

        # 幅度损失
        if self.use_amplitude:
            x_amp = torch.abs(x_fft)
            y_amp = torch.abs(y_fft)
            loss = loss + torch.mean(torch.abs(x_amp - y_amp))

        # 相位损失
        if self.use_phase:
            x_phase = torch.angle(x_fft)
            y_phase = torch.angle(y_fft)
            loss = loss + torch.mean(torch.abs(x_phase - y_phase))

        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失 (Perceptual Loss)

    使用预训练的 VGG 网络提取特征，计算特征空间的距离
    """
    def __init__(self, layer_weights=None, use_vgg16=True):
        super(PerceptualLoss, self).__init__()
        if layer_weights is None:
            # 默认使用 VGG16 的 relu3_3 层
            layer_weights = {'3': 1.0, '8': 1.0, '15': 1.0}

        self.layer_weights = layer_weights
        self.use_vgg16 = use_vgg16

        # 加载预训练 VGG
        if use_vgg16:
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        else:
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

        # 提取特征层
        self.feature_extractor = nn.ModuleList()
        layer_indices = sorted([int(k) for k in layer_weights.keys()])

        idx = 0
        for i, layer in enumerate(vgg.features):
            if idx >= len(layer_indices):
                break
            self.feature_extractor.append(layer)
            if i == layer_indices[idx]:
                idx += 1

        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """
        Args:
            x: 预测图像 [B, 3, H, W], 范围 [0, 1]
            y: 真值图像 [B, 3, H, W], 范围 [0, 1]
        Returns:
            loss: 标量
        """
        # 确保输入在 [0, 1] 范围
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)

        # ImageNet 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)

        x_norm = (x - mean) / std
        y_norm = (y - mean) / std

        # 提取特征
        x_features = []
        y_features = []

        idx = 0
        layer_indices = sorted([int(k) for k in self.layer_weights.keys()])

        for layer in self.feature_extractor:
            x_norm = layer(x_norm)
            y_norm = layer(y_norm)

            if idx < len(layer_indices) and isinstance(layer, nn.ReLU):
                x_features.append(x_norm)
                y_features.append(y_norm)
                idx += 1

        # 计算损失
        loss = 0.0
        for i, (x_feat, y_feat) in enumerate(zip(x_features, y_features)):
            layer_idx = layer_indices[i]
            weight = self.layer_weights[str(layer_idx)]
            loss = loss + weight * F.l1_loss(x_feat, y_feat)

        return loss


class ParameterSupervisionLoss(nn.Module):
    """
    物理参数监督损失

    当有物理参数真值时 (如去雾任务的透射率 T)，可以提供额外监督
    """
    def __init__(self, lambda_T=0.1, lambda_K=0.1, lambda_D=0.1):
        super(ParameterSupervisionLoss, self).__init__()
        self.lambda_T = lambda_T
        self.lambda_K = lambda_K
        self.lambda_D = lambda_D

    def forward(self, pred_params, gt_params):
        """
        Args:
            pred_params: 预测的参数字典
            gt_params: 真值参数字典 (可能为空)
        Returns:
            loss: 标量
        """
        loss = 0.0

        # T 监督
        if 'T' in gt_params and gt_params['T'] is not None:
            loss = loss + self.lambda_T * F.l1_loss(pred_params['T'], gt_params['T'])

        # K 监督 (模糊核)
        if 'K' in gt_params and gt_params['K'] is not None:
            loss = loss + self.lambda_K * F.l1_loss(pred_params['K'], gt_params['K'])

        # D 监督 (偏置)
        if 'D' in gt_params and gt_params['D'] is not None:
            loss = loss + self.lambda_D * F.l1_loss(pred_params['D'], gt_params['D'])

        return loss


class DFADUNLoss(nn.Module):
    """
    DFA-DUN 完整损失函数

    组合多个损失项:
    1. 空间域重建损失 (Charbonnier)
    2. 频域损失 (FFT)
    3. 感知损失 (可选)
    4. 参数监督损失 (可选)

    Args:
        lambda_rec: 重建损失权重
        lambda_freq: 频域损失权重
        lambda_percep: 感知损失权重
        lambda_param: 参数监督损失权重
        use_perceptual: 是否使用感知损失
    """
    def __init__(
        self,
        lambda_rec=1.0,
        lambda_freq=0.1,
        lambda_percep=0.05,
        lambda_param=0.0,
        use_parameter=False,
        use_perceptual=False
    ):
        super(DFADUNLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_freq = lambda_freq
        self.lambda_percep = lambda_percep
        self.lambda_param = lambda_param

        # 初始化各个损失
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.frequency = FrequencyLoss(use_amplitude=True, use_phase=False)
        
        self.use_parameter = use_parameter
        if use_parameter:
            self.param_supervision = ParameterSupervisionLoss(
                lambda_T=lambda_param,
                lambda_K=lambda_param,
                lambda_D=lambda_param
            )

        # 感知损失 (可选)
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = PerceptualLoss(
                layer_weights={'3': 0.5, '8': 0.5, '15': 0.5},
                use_vgg16=True
            )

    def forward(self, pred_img, gt_img, pred_params=None, gt_params=None):
        """
        Args:
            pred_img: 预测图像 [B, 3, H, W]
            gt_img: 真值图像 [B, 3, H, W]
            pred_params: 预测的参数 (列表或字典)
            gt_params: 真值参数 (可选)

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. 空间域重建损失
        loss_rec = self.charbonnier(pred_img, gt_img)
        loss_dict['loss_rec'] = loss_rec

        # 2. 频域损失
        loss_freq = self.frequency(pred_img, gt_img)
        loss_dict['loss_freq'] = loss_freq

        # 3. 感知损失 (可选)
        if self.use_perceptual:
            loss_percep = self.perceptual(pred_img, gt_img)
            loss_dict['loss_percep'] = loss_percep
        else:
            loss_percep = 0.0
            loss_dict['loss_percep'] = torch.tensor(0.0, device=pred_img.device)

        # 4. 参数监督损失 (可选)
        if gt_params is not None and pred_params is not None:
            # 如果 pred_params 是列表 (多阶段)，只监督最后一个阶段
            if isinstance(pred_params, list):
                params_to_supervise = pred_params[-1]
            else:
                params_to_supervise = pred_params

            loss_param = self.param_supervision(params_to_supervise, gt_params)
            loss_dict['loss_param'] = loss_param
        else:
            loss_param = 0.0
            loss_dict['loss_param'] = torch.tensor(0.0, device=pred_img.device)

        # 总损失
        total_loss = (
            self.lambda_rec * loss_rec +
            self.lambda_freq * loss_freq +
            self.lambda_percep * loss_percep +
            self.lambda_param * loss_param
        )

        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict


class SSIMLoss(nn.Module):
    """
    SSIM 损失 (Structural Similarity Index)

    衡量结构相似性，范围 [-1, 1]，越接近 1 越好
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)

    def forward(self, x, y):
        """
        Args:
            x: 预测图像 [B, 3, H, W]
            y: 真值图像 [B, 3, H, W]
        Returns:
            loss: 1 - SSIM (越小越好)
        """
        # 确保 C=3
        if x.shape[1] != self.channel:
            raise ValueError(f"Expected {self.channel} channels, got {x.shape[1]}")

        self.window = self.window.to(x.device)

        ssim_map = self._ssim(x, y)

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

    def _ssim(self, x, y):
        """
        计算 SSIM
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = self._gaussian_blur(x, self.window)
        mu_y = self._gaussian_blur(y, self.window)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = self._gaussian_blur(x * x, self.window) - mu_x_sq
        sigma_y_sq = self._gaussian_blur(y * y, self.window) - mu_y_sq
        sigma_xy = self._gaussian_blur(x * y, self.window) - mu_xy

        SSIM_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = SSIM_n / SSIM_d

        return ssim_map

    def _gaussian_blur(self, x, window):
        """
        高斯模糊
        """
        return F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)

    def _create_window(self, window_size, channel):
        """
        创建高斯窗口
        """
        def _gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()

        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window


# 测试代码
if __name__ == "__main__":
    print("测试损失函数...")

    # 创建测试数据
    B, C, H, W = 2, 3, 256, 256
    pred_img = torch.rand(B, C, H, W)
    gt_img = torch.rand(B, C, H, W)

    # 测试 Charbonnier Loss
    charbonnier = CharbonnierLoss()
    loss_char = charbonnier(pred_img, gt_img)
    print(f"Charbonnier Loss: {loss_char.item():.4f}")

    # 测试 Frequency Loss
    freq_loss = FrequencyLoss()
    loss_freq = freq_loss(pred_img, gt_img)
    print(f"Frequency Loss: {loss_freq.item():.4f}")

    # 测试 SSIM Loss
    ssim_loss = SSIMLoss()
    loss_ssim = ssim_loss(pred_img, gt_img)
    print(f"SSIM Loss: {loss_ssim.item():.4f}")

    # 测试完整 DFA-DUN Loss
    criterion = DFADUNLoss(
        lambda_rec=1.0,
        lambda_freq=0.1,
        lambda_percep=0.0,
        use_perceptual=False
    )

    total_loss, loss_dict = criterion(pred_img, gt_img)
    print(f"\nDFA-DUN Loss:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Reconstruction: {loss_dict['loss_rec'].item():.4f}")
    print(f"  Frequency: {loss_dict['loss_freq'].item():.4f}")

    print("\n所有损失函数测试通过!")
