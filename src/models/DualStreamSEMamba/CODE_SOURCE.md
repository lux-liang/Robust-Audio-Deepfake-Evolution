# Dual-Stream SE-Mamba 代码来源说明

本文档详细说明 Dual-Stream SE-Mamba 模型中每个模块的代码来源，确保所有代码都严格从原始代码库借用。

---

## 📋 模块来源清单

| 模块 | 原始文件 | 行号 | 状态 | 修改说明 |
|------|---------|------|------|---------|
| **CONV (SincConv)** | `models/AASIST.py` | 325-410 | ✅ 完全复制 | 无修改 |
| **Residual_block** | `models/AASIST.py` | 413-466 | ✅ 完全复制 | 无修改 |
| **SincNetEncoder** | 参考 `models/AASIST.py` | 469-607 | ⚠️ 适配修改 | 只提取前端部分，不包含 Graph Attention |
| **WavLMFrontend** | `models/MoEMambaASV.py` | 39-105 | ✅ 完全复制 | 无修改 |
| **PN_BiMambas_Encoder** | `Fake-Mamba-main/conformer00.py` | 327-458 | ✅ 完全复制 | 无修改 |
| **DualStreamFusion** | - | - | ⚠️ 必要适配 | 原始代码库无双流融合实现，进行必要适配 |
| **Model (主类)** | 参考 `models/MoEMambaASV.py` | 352-404 | ⚠️ 适配修改 | 整合双流架构 |

---

## 🔍 详细说明

### 1. CONV (SincConv)

**来源**: `models/AASIST.py` 第 325-410 行

**代码状态**: ✅ 完全复制，无任何修改

**验证方法**:
```bash
# 对比原始代码
diff models/AASIST.py:325-410 models/DualStreamSEMamba/DualStreamSEMamba.py:CONV类
```

---

### 2. Residual_block

**来源**: `models/AASIST.py` 第 413-466 行

**代码状态**: ✅ 完全复制，无任何修改

**验证方法**:
```bash
# 对比原始代码
diff models/AASIST.py:413-466 models/DualStreamSEMamba/DualStreamSEMamba.py:Residual_block类
```

---

### 3. SincNetEncoder

**来源**: 参考 `models/AASIST.py` 的 Model 类 (第 469-607 行)

**代码状态**: ⚠️ 适配修改

**修改说明**:
- **原始代码**: AASIST 的 Model 类包含完整的 SincNet 前端 + Graph Attention 后端
- **修改内容**: 只提取前端部分（CONV + encoder），不包含 Graph Attention
- **修改原因**: 我们需要 SincNet 作为前端特征提取器，而不是完整的 AASIST 模型
- **修改位置**: 
  - 保留: `conv_time`, `first_bn`, `selu`, `encoder` (第 479-494 行)
  - 移除: Graph Attention 相关代码 (第 496-526 行)
  - 修改: forward 方法只提取时域特征 `e_T`，不提取谱域特征 `e_S`

**代码对比**:
```python
# AASIST 原始代码 (第 528-550 行)
e = self.encoder(x)
e_S, _ = torch.max(torch.abs(e), dim=3)  # 谱域
e_T, _ = torch.max(torch.abs(e), dim=2)  # 时域
# ... Graph Attention ...

# 我们的修改
e = self.encoder(x)
e_T, _ = torch.max(torch.abs(e), dim=2)  # 只提取时域
e_T = e_T.transpose(1, 2)
return e_T
```

---

### 4. WavLMFrontend

**来源**: `models/MoEMambaASV.py` 第 39-105 行

**代码状态**: ✅ 完全复制，无任何修改

**验证方法**:
```bash
# 对比原始代码
diff models/MoEMambaASV.py:39-105 models/DualStreamSEMamba/DualStreamSEMamba.py:WavLMFrontend类
```

---

### 5. PN_BiMambas_Encoder

**来源**: `Fake-Mamba-main/conformer00.py` 第 327-458 行

**代码状态**: ✅ 完全复制，无任何修改

**关键代码** (从原始文件复制):
```python
# 完全按照原始代码实现
class PN_BiMambas_Encoder(nn.Module):
    def __init__(self, d_model, n_state):
        super(PN_BiMambas_Encoder, self).__init__()
        self.d_model = d_model
        self.mamba = Mamba(d_model, n_state)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)  # Pre-Norm
        mamba_out_forward = self.mamba(x_norm)
        x_flip = torch.flip(x_norm, dims=[1])
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])
        mamba_out = mamba_out_forward + mamba_out_backward
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)
        output = ff_out + residual
        return output
```

**验证方法**:
```bash
# 对比原始代码
diff /root/autodl-tmp/Fake-Mamba-main/Fake-Mamba-main/conformer00.py:327-458 \
     models/DualStreamSEMamba/DualStreamSEMamba.py:PN_BiMambas_Encoder类
```

---

### 6. DualStreamFusion

**来源**: ❌ 原始代码库中没有双流融合的实现

**代码状态**: ⚠️ 必要的适配修改

**修改说明**:
- **问题**: 原始代码库（AASIST、Fake-Mamba、MoEMambaASV）都没有双流融合的实现
- **解决方案**: 实现简单的线性投影+拼接融合
- **设计原则**: 最小化修改，只做必要的适配

**实现逻辑**:
```python
# 1. 投影对齐维度
f_w = Linear(1024 → emb_size)(f_wavlm)
f_s = Linear(64 → emb_size)(f_sinc)

# 2. 时间对齐
T = min(T1, T2)

# 3. 拼接融合
f_cat = concat([f_w, f_s], dim=-1)
f_fused = Linear(emb_size*2 → emb_size)(f_cat)
```

**为什么这样设计**:
- 简单有效: 不引入复杂的注意力机制
- 易于训练: 梯度流动顺畅
- 可解释: 模型可以学习两路特征的权重

---

### 7. Model (主类)

**来源**: 参考 `models/MoEMambaASV.py` 的 Model 类 (第 352-404 行)

**代码状态**: ⚠️ 适配修改

**修改说明**:
- **原始结构**: MoEMambaASV 只有单流（WavLM）
- **修改内容**: 添加 SincNet 流和融合模块
- **修改位置**:
  - 添加 `sinc_stream` 和 `fusion` 模块
  - 修改 `forward` 方法，处理双流输入

**代码对比**:
```python
# MoEMambaASV 原始代码
def forward(self, x):
    x_ssl = self.ssl_model(x)  # 单流
    x = self.LL(x_ssl)
    # ...

# 我们的修改
def forward(self, x):
    f_wavlm = self.wavlm_stream(x)  # 流1
    f_sinc = self.sinc_stream(x)    # 流2
    x = self.fusion(f_wavlm, f_sinc)  # 融合
    # ...
```

---

## ✅ 代码验证清单

在提交代码前，请确认：

- [x] CONV 类与 AASIST.py 完全一致
- [x] Residual_block 类与 AASIST.py 完全一致
- [x] WavLMFrontend 类与 MoEMambaASV.py 完全一致
- [x] PN_BiMambas_Encoder 类与 Fake-Mamba conformer00.py 完全一致
- [x] 所有适配修改都有详细说明
- [x] 没有引入原始代码库中不存在的复杂机制

---

## 📝 修改记录

| 日期 | 修改内容 | 原因 |
|------|---------|------|
| 2025-01-XX | 创建 DualStreamFusion 模块 | 原始代码库无双流融合实现 |
| 2025-01-XX | 修改 SincNetEncoder，只提取时域特征 | 适配双流架构需求 |
| 2025-01-XX | 修改 Model 主类，整合双流 | 适配双流架构需求 |

---

**文档版本**: v1.0  
**最后更新**: 2025-01-XX

