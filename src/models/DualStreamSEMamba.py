"""
Dual-Stream SE-Mamba: 双流融合架构

所有代码严格从原始代码库借用，仅做必要的适配修改。

代码来源:
========
1. SincConv & Residual_block: 
   - 来源: models/AASIST.py (第 325-466 行)
   - 状态: 完全复制，无修改

2. WavLM Frontend:
   - 来源: models/MoEMambaASV.py (第 39-105 行)
   - 状态: 完全复制，无修改

3. Pre-Norm BiMamba:
   - 来源: Fake-Mamba-main/conformer00.py (第 327-458 行)
   - 状态: 完全复制，无修改

4. 融合模块:
   - 说明: 由于原始代码库中没有双流融合的实现，此处使用简单的线性投影+拼接
   - 修改原因: 需要将 WavLM (1024-dim) 和 SincNet (64-dim) 特征融合
   - 修改内容: 添加投影层对齐维度，然后拼接

5. 主模型类:
   - 说明: 整合上述模块，参考 MoEMambaASV.py 的结构
   - 修改原因: 适配双流输入
"""

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os

# ============================================================
# Official Mamba imports
# ============================================================
from mamba_ssm.modules.mamba_simple import Mamba


# ============================================================
# 1. SincConv (从 AASIST.py 第 325-410 行完全复制)
# ============================================================
class CONV(nn.Module):
    """
    从 AASIST 官方代码完全复制
    来源: models/AASIST.py 第 325-410 行
    修改: 无
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


# ============================================================
# 2. Residual_block (从 AASIST.py 第 413-466 行完全复制)
# ============================================================
class Residual_block(nn.Module):
    """
    从 AASIST 官方代码完全复制
    来源: models/AASIST.py 第 413-466 行
    修改: 无
    """
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


# ============================================================
# 3. SincNet Encoder (封装 CONV + Residual_block)
# ============================================================
class SincNetEncoder(nn.Module):
    """
    封装 SincConv 和 Residual_block，参考 AASIST.py 的 Model 类实现
    
    来源: 参考 models/AASIST.py 第 469-607 行的前端部分
    修改说明:
    - 只提取 SincNet 前端部分（CONV + encoder），不包含 Graph Attention
    - 输出时域特征 (e_T)，用于后续融合
    """
    def __init__(self, sinc_channels=70, sinc_kernel=128):
        super().__init__()
        
        filts = [sinc_channels, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        # 从 AASIST 复制
        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=sinc_kernel,
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        # 从 AASIST 复制 encoder 结构
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        
        self.out_dim = filts[-1][-1]  # 64
    
    def forward(self, x, freq_aug=False):
        """
        参考 AASIST.py 的 forward 方法
        修改: 只提取时域特征 (e_T)，不提取谱域特征
        
        维度检查:
        - conv_time 输出: (B, 70, T)
        - unsqueeze(1) 后: (B, 1, 70, T)
        - max_pool2d(3,3) 后: (B, 1, 70/3, T/3) ≈ (B, 1, 23, T/3)
        - encoder 输出: (B, 64, F, T') 其中 F 是频域维度，T' 是时域维度
        - max along dim=2 (频域) 后: (B, 64, T')
        """
        x = x.unsqueeze(1)  # (B, 1, samples)
        x = self.conv_time(x, mask=freq_aug)  # (B, 70, T)
        x = x.unsqueeze(dim=1)  # (B, 1, 70, T)
        x = F.max_pool2d(torch.abs(x), (3, 3))  # (B, 1, 23, T/3)
        x = self.first_bn(x)
        x = self.selu(x)

        # get embeddings using encoder
        # 输出维度: (B, 64, F, T') 其中 F 是频域，T' 是时域
        e = self.encoder(x)
        
        # 维度检查：确保 e 是 4D (B, C, F, T)
        if e.ndim != 4:
            raise ValueError(f"Expected encoder output to be 4D (B, C, F, T), got {e.ndim}D with shape {e.shape}")
        
        # temporal GAT (GAT-T) - 只提取时域特征
        # max along dim=2 (频域维度) -> (B, 64, T')
        e_T, _ = torch.max(torch.abs(e), dim=2)  # (B, 64, T')
        e_T = e_T.transpose(1, 2)  # (B, T', 64)
        
        return e_T


# ============================================================
# 4. WavLM Frontend (从 MoEMambaASV.py 第 39-105 行完全复制)
# ============================================================
class WavLMFrontend(nn.Module):
    """
    标准 WavLM 前端封装 (Hugging Face 官方风格)
    支持动态冻结层数，兼容 Phase 4 和 Phase 5
    
    Args:
        model_path (str): 预训练模型名称或路径
        freeze_layers (int): 
            -1: 全量微调 (Full Fine-tuning)
             0: 冻结 CNN 前端，解冻所有 Transformer 层
             X: 冻结 CNN 前端 + 底层 X 层 Transformer (Phase 4=18, Phase 5=12)
    """
    def __init__(self, model_path="microsoft/wavlm-large", freeze_layers=18):
        super().__init__()
        from transformers import WavLMModel
        
        print(f"[WavLMFrontend] Initializing from {model_path}...")
        
        # 尝试从本地加载，如果失败则从 HuggingFace 下载
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        local_path1 = os.path.join(
            os.path.dirname(os.path.dirname(current_file_dir)),
            "pretrained", "microsoft", "wavlm-large"
        )
        work_dir = os.getcwd()
        local_path2 = os.path.join(work_dir, "pretrained", "microsoft", "wavlm-large")
        local_path3 = "/root/aasist-main/pretrained/microsoft/wavlm-large"
        
        local_path = None
        for path in [local_path1, local_path2, local_path3]:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "pytorch_model.bin")):
                local_path = path
                break
        
        if local_path:
            print(f"[WavLMFrontend] ✅ Loading from local: {local_path}")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            self.model = WavLMModel.from_pretrained(local_path, local_files_only=True)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
        else:
            print(f"[WavLMFrontend] ⚠️  Local model not found, loading from HuggingFace: {model_path}")
            self.model = WavLMModel.from_pretrained(model_path)
        
        self.out_dim = 1024
        
        # Enable outputting all hidden states for layer weighting
        self.model.config.output_hidden_states = True
        
        # Learnable weights for 25 layers (0-24)
        self.layer_weights = nn.Parameter(torch.zeros(25))
        
        # 1. 基础配置：开启梯度检查点 (节省 50% 显存，训练变慢 30%)
        # 这对于单卡跑 WavLM Large 是必须的
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("[WavLMFrontend] Gradient Checkpointing ENABLED.")
        else:
            print("[WavLMFrontend] Warning: gradient_checkpointing_enable not available.")
        
        # 2. 执行冻结策略 (核心逻辑)
        self.apply_freezing_strategy(freeze_layers)
    
    def apply_freezing_strategy(self, freeze_layers):
        """
        应用冻结策略 - 业界标准写法 (Hugging Face 官方模式)
        """
        # A. 始终冻结 CNN 特征提取器 (Feature Extractor)
        # 这部分负责从原始波形提取基础特征，极难训练且容易破坏
        self.model.feature_extractor.requires_grad_(False)
        self.model.feature_projection.requires_grad_(False)
        print("[WavLMFrontend] CNN Feature Extractor: FROZEN (Standard)")
        
        # B. 动态冻结 Transformer Encoder 层
        # WavLM Large 有 24 层 (索引 0-23)
        total_layers = len(self.model.encoder.layers)
        
        if freeze_layers < 0:
            # 策略: 全量微调 (Phase 5 激进版)
            print(f"[WavLMFrontend] Strategy: Full Fine-tuning (All {total_layers} layers trainable)")
            for layer in self.model.encoder.layers:
                layer.requires_grad_(True)
        else:
            # 策略: 部分微调 (Phase 4 & Phase 5 标准版)
            # freeze_layers = 18 -> 冻结 0-17，训练 18-23 (Phase 4)
            # freeze_layers = 12 -> 冻结 0-11，训练 12-23 (Phase 5，针对A18)
            print(f"[WavLMFrontend] Strategy: Freezing bottom {freeze_layers} / {total_layers} layers")
            
            for i, layer in enumerate(self.model.encoder.layers):
                if i < freeze_layers:
                    layer.requires_grad_(False)  # 冻结底层
                else:
                    layer.requires_grad_(True)   # 解冻高层
            
            trainable_layers = total_layers - freeze_layers
            print(f"[WavLMFrontend] Bottom {freeze_layers} layers frozen. Top {trainable_layers} layers trainable.")
    
    def train(self, mode=True):
        """
        Overridden train method to ensure WavLM CNN frontend and BNs stay frozen.
        Critical for fine-tuning stability with small batch sizes.
        """
        super().train(mode)
        
        if mode:
            # 1. Always freeze CNN feature extractor
            self.model.feature_extractor.eval()
            self.model.feature_projection.eval()
            
            # 2. Force freeze all BatchNorm layers in WavLM
            # Small batch sizes (e.g. 8) will wreck BN statistics
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
        
        return self

    def forward(self, x):
        """
        输入: x (Batch, Length) - 原始波形
        输出: features (Batch, Frames, 1024)
        """
        # WavLM expects (batch, samples)
        if x.ndim == 3:
            x = x.squeeze(-1)
        
        # WavLM 需要输入为 Float 类型
        if x.dtype != torch.float32:
            x = x.float()
        
        # 兼容 PeftModel (LoRA)
        # peft 包装后的 forward 会自动注入 input_ids 导致 WavLM 报错
        # 解决方案：直接调用内部的 base_model.model (即原始 WavLM)，
        # 因为 PEFT 是原地修改了层结构，直接调底层模型依然包含 LoRA 适配器。
        
        # 尝试导入 PeftModel 以进行类型检查
        try:
            from peft import PeftModel
            is_peft = isinstance(self.model, PeftModel)
        except ImportError:
            is_peft = False

        if is_peft:
            # 路径通常是: PeftModel -> LoraModel (base_model) -> WavLMModel (model)
            # 这样调用最安全，既避开了参数检查，又保留了 LoRA 计算
            outputs = self.model.base_model.model(x, output_hidden_states=True)
        else:
            # 普通模式
            outputs = self.model(x, output_hidden_states=True)
        
        # Layer Weighting Mechanism (保留原有加权机制)
        # hidden_states: tuple of 25 tensors (batch, frames, 1024)
        hidden_states = outputs.hidden_states
        
        # Stack to (25, batch, frames, 1024)
        stacked_states = torch.stack(hidden_states)
        
        # Softmax weights
        weights = F.softmax(self.layer_weights, dim=0)  # (25)
        
        # Weighted sum: sum(w_i * layer_i)
        # (25, 1, 1, 1) * (25, batch, frames, 1024) -> sum dim 0
        weighted_features = (weights.view(-1, 1, 1, 1) * stacked_states).sum(dim=0)
        
        return weighted_features


# ============================================================
# 5. Pre-Norm BiMamba (从 Fake-Mamba-main/conformer00.py 完全复制)
# ============================================================
class PN_BiMambas_Encoder(nn.Module):
    """
    从 Fake-Mamba-main 完全复制
    来源: Fake-Mamba-main/conformer00.py 第 327-458 行
    修改: 无
    """
    def __init__(self, d_model, n_state):
        super(PN_BiMambas_Encoder, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # ADD forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual
        return output


# ============================================================
# 6. SELayer (标准 Squeeze-and-Excitation 实现)
# ============================================================
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer (适配 1D 序列特征)
    
    说明: AASIST.py 中没有 SELayer 实现（只有 GraphAttentionLayer）
    此处使用标准的 SE 模块实现（参考 SENet 论文）
    
    设计原则:
    - Squeeze: 全局平均池化，压缩空间维度
    - Excitation: 两个全连接层，学习通道权重
    - Scale: 将权重应用到原始特征
    
    适配修改:
    - 原始 SE 用于 2D 特征 (B, C, H, W)，此处适配为 1D 序列 (B, T, C)
    - AdaptiveAvgPool2d -> AdaptiveAvgPool1d
    - 输入格式: (B, T, C) -> permute -> (B, C, T) -> pool -> (B, C)
    """
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 适配 1D
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, C) 序列特征
        Returns:
            x * se_weights: (B, T, C) 加权后的特征
        """
        b, t, c = x.size()
        # Permute to (B, C, T) for pooling
        y = x.permute(0, 2, 1)  # (B, C, T)
        y = self.avg_pool(y).view(b, c)  # (B, C)
        y = self.fc(y).view(b, 1, c)  # (B, 1, C)
        return x * y.expand_as(x)  # (B, T, C)


# ============================================================
# 7. 融合模块 (必要的适配修改)
# ============================================================
class DualStreamFusion(nn.Module):
    """
    双流特征融合模块
    
    说明: 原始代码库中没有双流融合的实现，此处进行必要的适配修改
    
    修改原因:
    - WavLM 输出: (batch, T1, 1024)
    - SincNet 输出: (batch, T2, 64)
    - 需要对齐时间维度和特征维度，然后融合
    
    修改内容:
    1. 投影层: 将两路特征投影到相同维度 (emb_size)
    2. 时间对齐: 使用 F.interpolate 插值对齐（修复时间错位问题）
    3. 拼接融合: 特征拼接 + 线性投影 + SE 注意力
    
    关键修复:
    - ❌ 原问题: 使用 min 截断会导致时间错位（WavLM 的 1 秒和 SincNet 的 0.1 秒拼在一起）
    - ✅ 修复: 使用 F.interpolate 将 SincNet 的时间维度插值到与 WavLM 一致
    """
    def __init__(self, wavlm_dim: int, sinc_dim: int, out_dim: int, reduction: int = 16):
        super().__init__()
        
        # === 关键修复：输入归一化 ===
        # 强制拉平两个流的特征分布，防止梯度被某一方主导
        # WavLM 特征值分布范围大，SincNet 经过 SELU+BN 分布受控
        # 如果不归一化，WavLM 的梯度会"淹没"SincNet，导致模型忽略 SincNet 分支
        self.ln_wavlm = nn.LayerNorm(wavlm_dim)
        self.ln_sinc = nn.LayerNorm(sinc_dim)
        
        # 投影到相同维度
        self.wavlm_proj = nn.Linear(wavlm_dim, out_dim)
        self.sinc_proj = nn.Linear(sinc_dim, out_dim)
        
        # 融合投影
        self.fusion_proj = nn.Linear(out_dim * 2, out_dim)
        
        # SE 注意力模块（标准 SE 实现，AASIST.py 中没有）
        self.se_layer = SELayer(out_dim, reduction=reduction)
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, f_wavlm: Tensor, f_sinc: Tensor) -> Tensor:
        """
        Args:
            f_wavlm: (B, T1, 1024) WavLM 特征
            f_sinc:  (B, T2, 64) SincNet 特征
        Returns:
            f_fused: (B, T1, out_dim) 融合后的特征
        """
        # === 关键修复：输入归一化 ===
        # 拉平两路特征的分布，防止梯度被某一方主导
        f_wavlm = self.ln_wavlm(f_wavlm)  # (B, T1, 1024)
        f_sinc = self.ln_sinc(f_sinc)      # (B, T2, 64)
        
        # 1. 投影
        f_w = self.wavlm_proj(f_wavlm)  # (B, T1, out_dim)
        f_s = self.sinc_proj(f_sinc)    # (B, T2, out_dim)
        
        # 2. 时间对齐（关键修复：使用插值而非截断）
        # 以 WavLM 的时间维度为准，将 SincNet 插值到相同长度
        # 注意：如果 SincNet 被压缩得太短（例如 4倍以上），考虑使用 'nearest' 模式保持边缘锐利度
        if f_s.size(1) != f_w.size(1):
            # Permute for interpolate: (B, T, C) -> (B, C, T)
            f_s = f_s.permute(0, 2, 1)  # (B, out_dim, T2)
            
            # 如果压缩比例过大（>4倍），使用 nearest 模式保持边缘锐利度
            # 这对于检测伪影很重要（边缘信息是关键）
            scale_factor = f_w.size(1) / f_s.size(2)
            if scale_factor > 4.0:
                # 使用 nearest 模式保持边缘锐利度
                f_s = F.interpolate(
                    f_s, 
                    size=f_w.size(1), 
                    mode='nearest'
                )  # (B, out_dim, T1)
            else:
                # 使用 linear 模式平滑插值
                f_s = F.interpolate(
                    f_s, 
                    size=f_w.size(1), 
                    mode='linear', 
                    align_corners=False
                )  # (B, out_dim, T1)
            
            # Permute back: (B, C, T) -> (B, T, C)
            f_s = f_s.permute(0, 2, 1)  # (B, T1, out_dim)
        
        # 3. 拼接融合
        f_cat = torch.cat([f_w, f_s], dim=-1)  # (B, T1, out_dim*2)
        f_fused = self.fusion_proj(f_cat)      # (B, T1, out_dim)
        
        # 4. SE 注意力（自适应特征加权）
        f_fused = self.se_layer(f_fused)  # (B, T1, out_dim)
        
        # 5. 归一化和 Dropout
        f_fused = self.norm(f_fused)
        f_fused = self.dropout(f_fused)
        
        return f_fused


# ============================================================
# 8. Main Model Class
# ============================================================
class Model(nn.Module):
    """
    Dual-Stream SE-Mamba Model
    
    架构:
    1. WavLM Stream (语义流) -> (B, T, 1024)
    2. SincNet Stream (信号流) -> (B, T', 64)
    3. DualStreamFusion -> (B, T, emb_size)
    4. Preprocessing (BN + SELU)
    5. Pre-Norm BiMamba Backbone -> (B, emb_size)
    6. Classifier -> (B, 2)
    
    参考结构: models/MoEMambaASV.py 的 Model 类
    """
    def __init__(self, args=None, device="cuda"):
        super().__init__()
        self.device = device
        
        # Hyperparameters
        emb_size = getattr(args, 'emb_size', 144) if args else 144
        num_encoders = getattr(args, 'num_encoders', 4) if args else 4
        d_state = getattr(args, 'd_state', 16) if args else 16
        sinc_channels = getattr(args, 'sinc_channels', 70) if args else 70
        
        print(f"\n{'='*60}")
        print(f"Dual-Stream SE-Mamba Model")
        print(f"{'='*60}")
        print(f"  Embedding Size: {emb_size}")
        print(f"  BiMamba Layers: {num_encoders}")
        print(f"  Mamba d_state: {d_state}")
        print(f"  SincNet Filters: {sinc_channels}")
        
        # ========== Stream 1: WavLM (语义流) ==========
        # === 关键修改：读取 freeze_layers 参数 ===
        # 优先从 args 读取 freeze_layers，如果没有则默认为 18 (Phase 4 设定)
        freeze_layers = getattr(args, 'wavlm_freeze_layers', 18) if args else 18
        self.wavlm_stream = WavLMFrontend(freeze_layers=freeze_layers)
        wavlm_dim = self.wavlm_stream.out_dim  # 1024
        
        # ========== Stream 2: SincNet (信号流) ==========
        self.sinc_stream = SincNetEncoder(sinc_channels=sinc_channels)
        sinc_dim = self.sinc_stream.out_dim  # 64
        
        # ========== Fusion ==========
        self.fusion = DualStreamFusion(
            wavlm_dim=wavlm_dim,
            sinc_dim=sinc_dim,
            out_dim=emb_size,
            reduction=16  # SE 模块的 reduction ratio
        )
        print(f"[Fusion] Dual-stream fusion: WavLM({wavlm_dim}) + SincNet({sinc_dim}) -> {emb_size} (with SE attention)")
        
        # ========== Pre-Norm BiMamba Backbone ==========
        # 注意: 不再需要 first_bn 和 selu，因为 DualStreamFusion 已经做了归一化
        self.backbone_layers = nn.ModuleList([
            PN_BiMambas_Encoder(d_model=emb_size, n_state=d_state)
            for _ in range(num_encoders)
        ])
        
        # Final norm and pooling
        self.norm_f = nn.LayerNorm(emb_size)
        self.attention_pool = nn.Linear(emb_size, 1)
        self.dropout = nn.Dropout(0.1)
        
        print(f"[Backbone] Pre-Norm BiMamba: {num_encoders} layers, d_state={d_state}")
        
        # ========== Classifier ==========
        self.classifier = nn.Linear(emb_size, 2)
        
        # 打印参数统计
        self._print_params()
        print(f"{'='*60}\n")
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        wavlm_params = sum(p.numel() for p in self.wavlm_stream.parameters())
        sinc_params = sum(p.numel() for p in self.sinc_stream.parameters())
        
        print(f"[Params] Total: {total:,}")
        print(f"[Params] Trainable: {trainable:,}")
        print(f"[Params] WavLM: {wavlm_params:,} (mostly frozen)")
        print(f"[Params] SincNet: {sinc_params:,} (all trainable)")
    
    def forward(self, x, Freq_aug=False):
        """
        Forward pass.
        
        Args:
            x: (batch, samples) 原始波形
            Freq_aug: 是否进行频率增强
        
        Returns:
            features: (batch, emb_size) 用于 OCSoftmax
            output: (batch, 2) 分类 logits
        """
        if x.ndim == 3:
            x = x.squeeze(-1)
        
        # ========== Stream 1: WavLM ==========
        f_wavlm = self.wavlm_stream(x)  # (batch, T1, 1024)
        
        # ========== Stream 2: SincNet ==========
        f_sinc = self.sinc_stream(x, freq_aug=Freq_aug)  # (batch, T2, 64)
        
        # ========== Fusion ==========
        f_fused = self.fusion(f_wavlm, f_sinc)  # (batch, T, emb_size)
        
        # ========== Pre-Norm BiMamba ==========
        # 注意: DualStreamFusion 已经做了 LayerNorm，不需要额外的 BatchNorm2d
        # 直接进入 Backbone 即可
        for layer in self.backbone_layers:
            f_fused = layer(f_fused)
        
        # Final norm
        f_fused = self.norm_f(f_fused)
        
        # Attention pooling
        attn_weights = F.softmax(self.attention_pool(f_fused), dim=1)  # (B, T, 1)
        features = torch.matmul(attn_weights.transpose(1, 2), f_fused).squeeze(1)  # (B, emb_size)
        features = self.dropout(features)
        
        # ========== Classifier ==========
        output = self.classifier(features)  # (batch, 2)
        
        return features, output

