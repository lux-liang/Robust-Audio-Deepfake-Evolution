"""
MoE-Mamba-ASV: WavLM + SincNet 双流前端 + MoE-Mamba Backbone

架构演进:
=========
v1 (Epoch 38): WavLM-only → TTS检测极强 (A07-A16 <1.5%), 但VC检测弱 (A17-A19 ~20-40%)
v2 (当前):     WavLM + SincNet → 双流融合，目标弥补VC检测短板

为什么需要双流?
==============
问题: WavLM 预训练时学会了"去噪"，会把声码器伪影当作噪声滤除
     → A17/A18/A19 (VC+Vocoder) 的高频伪影被"盲"掉了

解决: 添加 SincNet 流 (从 AASIST 借用)
     → SincNet 的可学习带通滤波器能检测这些伪影

架构:
=====
             ┌─────────────────┐
             │   Raw Waveform  │
             └────────┬────────┘
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
┌─────────────┐               ┌─────────────┐
│   WavLM     │               │   SincNet   │
│  (语义流)    │               │  (信号流)    │
│  1024-dim   │               │   64-dim    │
└──────┬──────┘               └──────┬──────┘
       │                             │
       ▼                             ▼
  Linear(144)                  Linear(144)
       │                             │
       └──────────┬──────────────────┘
                  │ Gated Fusion
                  ▼
         ┌─────────────────┐
         │   MoE-Mamba     │
         │  (4 Experts)    │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │   Classifier    │
         └─────────────────┘

参考:
====
- AASIST (ICASSP 2022): SincConv + ResBlocks 设计
- XLSR-Mamba: BiMamba 设计
"""

import random
import math
from functools import partial
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os

# ============================================================
# Official Mamba imports
# ============================================================
from mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ============================================================
# 1. WavLM Frontend (语义流 - Semantic Stream)
# ============================================================
class WavLMFrontend(nn.Module):
    """
    WavLM-Large 语义流前端
    - 25层加权融合 (Learnable Weighted Sum)
    - Bottom 18 冻结, Top 6 可训练
    """
    def __init__(self, model_path="microsoft/wavlm-large", freeze=True):
        super().__init__()
        from transformers import WavLMModel
        
        local_path = "/root/aasist-main/pretrained/microsoft/wavlm-large"
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "pytorch_model.bin")):
            print(f"[WavLM Stream] Loading from local: {local_path}")
            self.model = WavLMModel.from_pretrained(local_path)
        else:
            print(f"[WavLM Stream] Loading from HuggingFace: {model_path}")
            self.model = WavLMModel.from_pretrained(model_path)
        
        self.out_dim = 1024  
        self.model.config.output_hidden_states = True
        self.layer_weights = nn.Parameter(torch.zeros(25)) 
        
        if freeze:
            self.model.feature_extractor.requires_grad_(False)
            self.model.feature_projection.requires_grad_(False)
            
            for i, layer in enumerate(self.model.encoder.layers):
                if i < 18:
                    layer.requires_grad_(False)
            
            print("[WavLM Stream] Bottom 18 layers frozen, Top 6 trainable")
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(-1)
        
        outputs = self.model(x)
        hidden_states = outputs.hidden_states 
        stacked_states = torch.stack(hidden_states)
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_features = (weights.view(-1, 1, 1, 1) * stacked_states).sum(dim=0)
        
        return weighted_features


# ============================================================
# 2. SincNet Frontend (信号流 - Signal Stream)
#    从 AASIST 官方代码借用: CONV + Residual_block
# ============================================================
class SincConv(nn.Module):
    """
    可学习的带通滤波器组 (从 AASIST 官方代码借用)
    
    原理:
    - 初始化为 Mel 频率间隔的带通滤波器
    - 训练时可以学习调整滤波器的中心频率和带宽
    - 能自适应检测声码器在不同频带留下的痕迹 (特别是 4-8kHz)
    
    参数:
    - out_channels: 滤波器数量 (默认 70)
    - kernel_size: 滤波器长度 (默认 128)
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels: int = 70,
                 kernel_size: int = 128,
                 sample_rate: int = 16000,
                 in_channels: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 groups: int = 1):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel")
        if bias:
            raise ValueError('SincConv does not support bias')
        if groups > 1:
            raise ValueError('SincConv does not support groups')

        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Mel 频率初始化
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
            hHigh = (2 * fmax / self.sample_rate) * \
                np.sinc(2 * fmax * self.hsupp.numpy() / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * \
                np.sinc(2 * fmin * self.hsupp.numpy() / self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask: bool = False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        
        if mask and self.training:
            # Frequency Augmentation: 随机遮蔽部分滤波器
            A = random.randint(0, 20)
            A0 = random.randint(0, max(1, band_pass_filter.shape[0] - A - 1))
            band_pass_filter[A0:A0 + A, :] = 0

        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, bias=None, groups=1)


class Residual_block(nn.Module):
    """
    残差卷积块 (从 AASIST 官方代码借用)
    
    作用: 在 SincConv 输出的 (频带, 时间) 2D 表示上提取时频特征
    """
    def __init__(self, nb_filts: List[int], first: bool = False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(0, 1), stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                                             padding=(0, 1), kernel_size=(1, 3), stride=1)
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


class SincNetEncoder(nn.Module):
    """
    完整的 SincNet 编码器 (封装 SincConv + ResBlocks)
    
    输入: (batch, samples) 原始波形
    输出: (batch, T, 64) 时频特征
    """
    def __init__(self, sinc_channels: int = 70, sinc_kernel: int = 128):
        super().__init__()
        
        filts = [sinc_channels, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.sinc_conv = SincConv(out_channels=sinc_channels, kernel_size=sinc_kernel)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        self.encoder = nn.Sequential(
            Residual_block(nb_filts=filts[1], first=True),
            Residual_block(nb_filts=filts[2]),
            Residual_block(nb_filts=filts[3]),
            Residual_block(nb_filts=filts[4]),
            Residual_block(nb_filts=filts[4]),
            Residual_block(nb_filts=filts[4])
        )
        
        self.out_dim = filts[4][1]  # 64
        print(f"[SincNet Stream] {sinc_channels} filters -> {self.out_dim}-dim output")
    
    def forward(self, x, freq_aug: bool = False):
        # (batch, samples) -> (batch, 1, samples)
        x = x.unsqueeze(1)
        
        # SincConv: (batch, 1, samples) -> (batch, 70, T)
        x = self.sinc_conv(x, mask=freq_aug)
        
        # Add channel: (batch, 70, T) -> (batch, 1, 70, T)
        x = x.unsqueeze(1)
        
        # MaxPool + BN + SELU
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        
        # ResBlocks: (batch, 1, 23, T') -> (batch, 64, ~, T'')
        x = self.encoder(x)
        
        # 提取时域特征: max along freq -> (batch, 64, T)
        e_T, _ = torch.max(torch.abs(x), dim=2)
        e_T = e_T.transpose(1, 2)  # (batch, T, 64)
        
        return e_T


# ============================================================
# 3. MoE Components
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, mult=4, dropout=0.0, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class SwitchGate(nn.Module):
    def __init__(self, dim, num_experts: int, top_k: int = 2, 
                 capacity_factor: float = 1.0, epsilon: float = 1e-6, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        gate_logits = self.w_gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(2, top_k_indices, 1.0)
        masked_gate_scores = gate_scores * mask
        sums = masked_gate_scores.sum(dim=-1, keepdim=True) + self.epsilon
        final_gate_scores = masked_gate_scores / sums
        return final_gate_scores, None 


class SwitchMoE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, output_dim: int, 
                 num_experts: int = 4, capacity_factor: float = 1.0,
                 mult: int = 4, use_aux_loss: bool = False, top_k: int = 2, *args, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.use_aux_loss = use_aux_loss
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim, mult=mult) for _ in range(num_experts)])
        self.gate = SwitchGate(dim, num_experts, top_k, capacity_factor)

    def forward(self, x: Tensor):
        gate_scores, _ = self.gate(x, use_aux_loss=self.use_aux_loss)
        if torch.isnan(gate_scores).any():
            gate_scores[torch.isnan(gate_scores)] = 0

        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = gate_scores[:, :, i].unsqueeze(-1)
            final_output += weight * expert_out
            
        return final_output, None


# ============================================================
# 4. MoE-Mamba Backbone
# ============================================================
def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, 
                 residual_in_fp32=False, fused_add_norm=False, layer_idx=None, device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls,
                  fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


class MoEMixerModel(nn.Module):
    """
    双向 Mamba + MoE 后端
    结构: [BiMamba Block -> MoE Layer] x N
    """
    def __init__(self, d_model: int, n_layer: int, num_experts: int = 4, top_k: int = 2,
                 ssm_cfg=None, norm_epsilon: float = 1e-5, rms_norm: bool = False,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.forward_layers = nn.ModuleList()
        self.forward_moe = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        self.backward_moe = nn.ModuleList()
        
        for i in range(n_layer):
            self.forward_layers.append(create_block(d_model, ssm_cfg, norm_epsilon, rms_norm, layer_idx=i, **factory_kwargs))
            self.forward_moe.append(SwitchMoE(d_model, d_model*4, output_dim=d_model, num_experts=num_experts, top_k=top_k))
            self.backward_layers.append(create_block(d_model, ssm_cfg, norm_epsilon, rms_norm, layer_idx=i, **factory_kwargs))
            self.backward_moe.append(SwitchMoE(d_model, d_model*4, output_dim=d_model, num_experts=num_experts, top_k=top_k))
            
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)
        self.f_attention_pool = nn.Linear(d_model, 1)
        self.b_attention_pool = nn.Linear(d_model, 1)
        self.LL = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        hidden_states = self.dropout(x)
        
        # Forward Path
        f_states, f_residual = hidden_states, None
        for mamba, moe in zip(self.forward_layers, self.forward_moe):
            f_states, f_residual = mamba(f_states, f_residual)
            curr_states = (f_states + f_residual) if f_residual is not None else f_states
            moe_out, _ = moe(curr_states)
            f_residual = f_residual + moe_out 
        
        f_residual = (f_states + f_residual) if f_residual is not None else f_states
        f_states = self.norm_f(f_residual.to(dtype=self.norm_f.weight.dtype))

        # Backward Path
        b_states, b_residual = hidden_states.flip([1]), None
        for mamba, moe in zip(self.backward_layers, self.backward_moe):
            b_states, b_residual = mamba(b_states, b_residual)
            curr_states = (b_states + b_residual) if b_residual is not None else b_states
            moe_out, _ = moe(curr_states)
            b_residual = b_residual + moe_out

        b_residual = (b_states + b_residual) if b_residual is not None else b_states
        b_states = self.norm_f(b_residual.to(dtype=self.norm_f.weight.dtype))
        
        # Pooling
        f_pool = torch.matmul(F.softmax(self.f_attention_pool(f_states), dim=1).transpose(-1, -2), f_states).squeeze(-2)
        b_pool = torch.matmul(F.softmax(self.b_attention_pool(b_states), dim=1).transpose(-1, -2), b_states).squeeze(-2)
        
        combined = torch.cat((f_pool, b_pool), dim=1)
        combined = self.LL(combined)
        combined = self.dropout(combined)
        
        return combined


# ============================================================
# 5. Main Model Class
# ============================================================
class Model(nn.Module):
    """
    MoE-Mamba-ASV v2: 双流融合架构
    
    新增参数:
    - use_sinc_stream: 是否启用 SincNet 流 (默认 False 保持向后兼容)
    - sinc_channels: SincConv 滤波器数量
    
    当 use_sinc_stream=True 时:
    - 同时使用 WavLM (语义) 和 SincNet (信号) 两个前端
    - 通过门控融合策略合并两路特征
    """
    def __init__(self, args=None, device="cuda"):
        super().__init__()
        self.device = device
        
        # Hyperparameters
        emb_size = getattr(args, 'emb_size', 144) if args else 144
        num_encoders = getattr(args, 'num_encoders', 6) if args else 6
        num_experts = getattr(args, 'num_experts', 4) if args else 4
        top_k = getattr(args, 'top_k', 2) if args else 2
        
        # 新增: 双流开关
        self.use_sinc_stream = getattr(args, 'use_sinc_stream', False) if args else False
        sinc_channels = getattr(args, 'sinc_channels', 70) if args else 70
        
        print(f"\n{'='*60}")
        print(f"MoE-Mamba-ASV {'v2 (Dual-Stream)' if self.use_sinc_stream else 'v1 (WavLM-only)'}")
        print(f"{'='*60}")
        print(f"  Embedding Size: {emb_size}")
        print(f"  Layers: {num_encoders} (Bidirectional)")
        print(f"  Experts: {num_experts} (Top-{top_k})")
        print(f"  SincNet Stream: {'Enabled' if self.use_sinc_stream else 'Disabled'}")
        
        # ========== Stream 1: WavLM (语义流) ==========
        self.ssl_model = WavLMFrontend(freeze=True)
        self.wavlm_proj = nn.Linear(self.ssl_model.out_dim, emb_size)
        
        # ========== Stream 2: SincNet (信号流) - 可选 ==========
        if self.use_sinc_stream:
            self.sinc_model = SincNetEncoder(sinc_channels=sinc_channels)
            self.sinc_proj = nn.Linear(self.sinc_model.out_dim, emb_size)
        
            # 门控融合网络
            self.fusion_gate = nn.Sequential(
                nn.Linear(emb_size * 2, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.Sigmoid()
            )
            self.fusion_norm = nn.LayerNorm(emb_size)
        
        # ========== Preprocessing ==========
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        
        # ========== MoE-Mamba Backbone ==========
        self.backbone = MoEMixerModel(
            d_model=emb_size,
            n_layer=num_encoders // 2,
            num_experts=num_experts,
            top_k=top_k,
            device=device
        )
        
        # ========== Classifier ==========
        self.classifier = nn.Linear(emb_size, 2)
        
        # 打印参数统计
        self._print_params()
        print(f"{'='*60}\n")
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total Params: {total:,}")
        print(f"  Trainable: {trainable:,}")
        if self.use_sinc_stream:
            sinc_params = sum(p.numel() for p in self.sinc_model.parameters())
            print(f"  SincNet Params: {sinc_params:,} (all trainable)")
        
    def forward(self, x, Freq_aug=False):
        """
        前向传播
        
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
        f_wavlm = self.ssl_model(x)  # (batch, T1, 1024)
        f_wavlm = self.wavlm_proj(f_wavlm)  # (batch, T1, emb_size)
        
        # ========== Stream 2: SincNet (如果启用) ==========
        if self.use_sinc_stream:
            f_sinc = self.sinc_model(x, freq_aug=Freq_aug)  # (batch, T2, 64)
            f_sinc = self.sinc_proj(f_sinc)  # (batch, T2, emb_size)
            
            # 对齐时间维度
            T = min(f_wavlm.size(1), f_sinc.size(1))
            f_wavlm = f_wavlm[:, :T, :]
            f_sinc = f_sinc[:, :T, :]
            
            # 门控融合
            gate_input = torch.cat([f_wavlm, f_sinc], dim=-1)
            gate = self.fusion_gate(gate_input)
            x = gate * f_wavlm + (1 - gate) * f_sinc
            x = self.fusion_norm(x)
        else:
            x = f_wavlm
        
        # ========== Preprocessing ==========
        x = x.unsqueeze(1)  # (batch, 1, T, emb_size)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(1)  # (batch, T, emb_size)
        
        # ========== MoE-Mamba Backbone ==========
        features = self.backbone(x)  # (batch, emb_size)
        
        # ========== Classifier ==========
        output = self.classifier(features)  # (batch, 2)
        
        return features, output
