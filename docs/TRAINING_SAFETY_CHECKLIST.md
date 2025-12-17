# 训练安全检查清单 (Training Safety Checklist)

本文档列出所有已实施的防御措施，确保模型训练稳定且不会过拟合。

---

## ✅ 已完成的防御措施

### 1. 梯度爆炸防御 (Gradient Explosion Prevention)

#### ✅ 防线 A: Pre-Norm 结构
- **位置**: `PN_BiMambas_Encoder` 类
- **状态**: ✅ 已实现
- **说明**: 在进入 Mamba 模块前先做 LayerNorm，这是训练深层序列模型的标准做法

#### ✅ 防线 B: 输入特征归一化
- **位置**: `DualStreamFusion.__init__()` 和 `forward()`
- **状态**: ✅ 已实现
- **代码**:
  ```python
  self.ln_wavlm = nn.LayerNorm(wavlm_dim)
  self.ln_sinc = nn.LayerNorm(sinc_dim)
  # 在 forward 中：
  f_wavlm = self.ln_wavlm(f_wavlm)
  f_sinc = self.ln_sinc(f_sinc)
  ```
- **作用**: 拉平 WavLM 和 SincNet 的特征分布，防止梯度被某一方主导

#### ✅ 防线 C: 梯度裁剪
- **位置**: `main.py` 第 521 行
- **状态**: ✅ 已实现
- **代码**:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
  ```
- **作用**: 防止训练中途 Loss 变成 NaN 的最后一道防线

---

### 2. 过拟合防御 (Overfitting Prevention)

#### ✅ 防线 A: WavLM 分层冻结
- **位置**: `WavLMFrontend.__init__()`
- **状态**: ✅ 已实现
- **策略**: 
  - Bottom 18 层冻结 (`requires_grad=False`)
  - Top 6 层可训练 (`requires_grad=True`)
- **说明**: 如果发现过拟合，可以将冻结层数从 18 改为 22 或 24（全冻结）

#### ✅ 防线 B: RawBoost 数据增强
- **位置**: `data_utils.py` 和 `config/DualStreamSEMamba.conf`
- **状态**: ✅ 已实现
- **配置**:
  ```json
  "data_config": {
      "rawboost_algo": 5  // 压缩编码增强，专治 A19
  }
  ```
- **说明**: 
  - `algo=5`: 随机选择 LNL/ISD/SSI/LNL+ISD 中的一种
  - 50% 概率触发增强
  - 模拟 MP3/AAC 压缩、电话信道等恶劣环境

#### ✅ 防线 C: 权重衰减 (Weight Decay)
- **位置**: `main.py` 第 207 行
- **状态**: ✅ 已实现
- **配置**:
  ```python
  weight_decay=optim_config['weight_decay']  # 默认 0.0001
  ```
- **说明**: 约束参数不乱跑，防止过拟合

#### ✅ 防线 D: 梯度检查点 (Gradient Checkpointing)
- **位置**: `WavLMFrontend.__init__()`
- **状态**: ✅ 已实现
- **作用**: 
  - 节省显存约 40-50%
  - 代价是前向传播时间增加约 20-30%
  - 可以支持更大的 batch size

---

## 📋 训练前最终检查清单

在开始训练前，请确认以下所有项目：

### 代码检查
- [x] **Pre-Norm**: `PN_BiMambas_Encoder` 使用 Pre-Norm 结构
- [x] **输入归一化**: `DualStreamFusion` 中对 WavLM 和 SincNet 分别做 LayerNorm
- [x] **梯度裁剪**: `main.py` 中 `clip_grad_norm_(..., max_norm=3.0)`
- [x] **数据增强**: `config/DualStreamSEMamba.conf` 中 `rawboost_algo: 5`
- [x] **权重衰减**: `optim_config.weight_decay: 0.0001`
- [x] **梯度检查点**: `WavLMFrontend` 中启用 `gradient_checkpointing_enable()`

### 环境检查
- [ ] **Mamba 环境**: `import mamba_ssm` 无报错，CUDA 可用
- [ ] **RawBoost 路径**: `from rawboost import RawBoost` 无报错
- [ ] **数据集路径**: 确认 `database_path` 正确指向 LA 数据集

### 关键实战检查（容易被忽视）
- [x] **DataLoader drop_last**: 训练集 `drop_last=True`（防止 BatchNorm/LayerNorm 在 batch_size=1 时出错）
- [x] **随机种子控制**: `set_seed()` 设置了所有随机数生成器（Python, NumPy, PyTorch, CUDA）
- [x] **RawBoost 随机性**: RawBoost 使用 `np.random`，已通过 `np.random.seed()` 控制
- [x] **CUDNN 确定性**: `cudnn.deterministic=True` 确保 CUDA 操作可复现

### 配置检查
- [ ] **Batch Size**: 根据显存调整（建议 8-16）
- [ ] **Learning Rate**: `base_lr: 0.00005`（已优化）
- [ ] **WavLM 冻结**: 确认打印信息显示 "Bottom 18 layers frozen"

---

## 🚨 训练监控指标

训练过程中，密切关注以下指标：

### 正常训练指标
- **Loss**: 应该平稳下降，不应出现突然跳跃
- **Dev EER**: 应该逐步下降
- **梯度范数**: 应该在 0.1-3.0 范围内（通过 `clip_grad_norm_` 控制）

### 异常信号（需要立即停止）
- ❌ **Loss 突然变成 NaN**: 检查梯度裁剪是否生效
- ❌ **Loss 不下降**: 检查学习率是否过小
- ❌ **Dev EER 上升但 Train Loss 下降**: 过拟合，考虑增加冻结层数
- ❌ **显存溢出 (OOM)**: 减小 batch size 或启用梯度检查点

---

## 🚨 补充检查项（实战中容易被忽视）

### 隐患 1: DataLoader 的 `drop_last` 参数

**风险**: 
- 使用 `BatchNorm` 或 `LayerNorm` 时，如果最后一个 Batch 只有 1 个样本（Batch Size=1），会导致标准差计算出错或 Loss NaN
- 对于我们的模型，`DualStreamFusion` 中使用了 `LayerNorm`，必须确保每个 batch 至少有 2 个样本

**状态**: ✅ 已修复
- **位置**: `main.py` 第 388 行
- **设置**: `drop_last=True`（训练集）
- **说明**: 验证集和测试集使用 `drop_last=False` 是合理的，因为评估时需要所有数据

### 隐患 2: RawBoost 的随机性控制

**风险**: 
- RawBoost 使用 `np.random` 进行随机增强
- 如果随机种子没固定好，会导致：
  1. 训练结果无法复现
  2. 验证集评估时因为随机噪声导致 EER 抖动剧烈
  3. 不同运行之间的结果不一致

**状态**: ✅ 已修复
- **位置**: `utils.py` 的 `set_seed()` 函数
- **修复内容**:
  ```python
  random.seed(seed)      # Python random (RawBoost 可能使用)
  np.random.seed(seed)   # NumPy random (RawBoost 使用)
  torch.manual_seed(seed)  # PyTorch random
  torch.cuda.manual_seed_all(seed)  # CUDA random
  torch.backends.cudnn.deterministic = True  # 确保 CUDA 操作可复现
  ```
- **说明**: 
  - `set_seed()` 在 `main.py` 第 61 行被调用
  - DataLoader 使用 `generator=gen` 和 `worker_init_fn=seed_worker` 确保每个 worker 的随机性也被控制

---

## 🔧 故障排除

### 问题 1: Loss 变成 NaN
**解决方案**:
1. 检查梯度裁剪是否生效（`max_norm=3.0`）
2. 检查输入归一化是否正确应用
3. 降低学习率（例如从 5e-5 降到 1e-5）

### 问题 2: 过拟合（Dev EER 上升）
**解决方案**:
1. 增加 WavLM 冻结层数（从 18 改为 22 或 24）
2. 增加 RawBoost 增强概率（从 50% 改为 80%）
3. 增加 weight_decay（从 0.0001 改为 0.0005）

### 问题 3: 显存不足 (OOM)
**解决方案**:
1. 减小 batch size（从 12 改为 8）
2. 确认梯度检查点已启用
3. 使用混合精度训练（AMP，已在代码中）

---

## 📊 预期训练曲线

### 正常训练曲线
- **Epoch 0-10**: Loss 快速下降，Dev EER 快速下降
- **Epoch 10-30**: Loss 缓慢下降，Dev EER 缓慢下降
- **Epoch 30-50**: Loss 趋于平稳，Dev EER 趋于平稳

### 目标性能
- **Dev EER**: < 2.0% (Phase 3 是 1.139%)
- **Eval EER**: < 10% (Phase 3 是 9.17%)
- **A19 EER**: < 5% (Phase 3 是 23%，目标是显著改善)

---

**文档版本**: v1.0  
**最后更新**: 2025-01-XX  
**状态**: ✅ 所有防御措施已实施

