# 项目文件结构说明

本文档说明整理后的项目文件结构，所有模型相关文件按模型分类组织。

---

## 📁 目录结构

```
/root/aasist-main/
├── models/                          # 模型代码目录
│   ├── AASIST.py                    # AASIST 原始模型（参考用）
│   ├── RawNet2Spoof.py             # RawNet2 基线模型
│   ├── RawNetGatSpoofST.py         # RawGAT-ST 基线模型
│   ├── MoEMambaASV.py              # ⭐ Phase 3: MoE-Mamba-ASV (当前主力)
│   ├── DualStreamSEMamba.py        # ⭐ Phase 4: Dual-Stream SE-Mamba (新模型)
│   ├── modules/                     # 共享模块
│   ├── official/                    # 官方参考实现
│   └── weights/                     # 模型权重
│
├── config/                          # 配置文件目录
│   ├── AASIST.conf                  # AASIST 配置
│   ├── AASIST-L.conf                # AASIST-L 配置
│   ├── RawNet2_baseline.conf        # RawNet2 配置
│   ├── RawGATST_baseline.conf       # RawGAT-ST 配置
│   ├── MoEMambaASV.conf             # ⭐ Phase 3 配置
│   └── DualStreamSEMamba.conf       # ⭐ Phase 4 配置
│
├── exp_result/                      # 实验结果目录
│   ├── baseline_best/               # 基线最佳模型
│   ├── LA_AASIST_ep100_bs24/        # AASIST 训练结果
│   ├── MoE-Mamba-ASV_*/             # ⭐ Phase 3 训练结果
│   └── DualStreamSEMamba_*/         # ⭐ Phase 4 训练结果（待生成）
│
├── models/MoEMambaASV/              # Phase 3 相关文档
│   └── CODE_SOURCE.md               # 代码来源说明
│
├── models/DualStreamSEMamba/        # Phase 4 相关文档
│   ├── CODE_SOURCE.md               # 代码来源说明
│   └── __init__.py
│
├── backup_models/                   # 模型备份
│   ├── best_model_phase2.5_epoch23.pth
│   └── best_model_phase2.5_epoch33.pth
│
├── LA/                              # ASVspoof 2019 LA 数据集
├── pretrained/                      # 预训练模型
│   └── microsoft/
│       └── wavlm-large/
│
├── main.py                          # 主训练脚本
├── evaluation.py                    # 评估脚本
├── data_utils.py                    # 数据处理工具
├── loss.py                          # 损失函数
├── utils.py                         # 工具函数
│
├── MODEL_EVOLUTION_DETAILED_REPORT.md  # ⭐ 模型迭代历程报告
├── DUAL_STREAM_SE_MAMBA_MODULE_GUIDE.md # ⭐ 模块整合指南
└── PROJECT_STRUCTURE.md             # 本文档
```

---

## 🎯 模型分类说明

### Phase 1-2: 已淘汰模型

**状态**: ❌ 已删除

**原因**: 
- Phase 1 (Cascade-Mamba): 过度设计，训练不稳定
- Phase 2 (WavLM-Mamba): 被 Phase 3 替代

**相关文件**: 已清理

---

### Phase 3: MoE-Mamba-ASV (当前主力)

**模型文件**: `models/MoEMambaASV.py`

**配置文件**: `config/MoEMambaASV.conf`

**训练结果**: `exp_result/MoE-Mamba-ASV_*/`

**最佳成绩**: 
- Dev EER: 1.139% (Epoch 38)
- Eval EER: 9.17%
- min t-DCF: 0.1519

**特点**:
- WavLM 前端 + MoE-Mamba 后端
- 4 个专家，Top-2 路由
- OC-Softmax 损失

**文档**: `models/MoEMambaASV/CODE_SOURCE.md`

---

### Phase 4: Dual-Stream SE-Mamba (新模型)

**模型文件**: `models/DualStreamSEMamba.py`

**配置文件**: `config/DualStreamSEMamba.conf`

**训练结果**: `exp_result/DualStreamSEMamba_*/` (待生成)

**特点**:
- 双流前端: WavLM (语义) + SincNet (信号)
- Pre-Norm BiMamba 后端
- 简单线性融合（无 SE，避免创新）

**代码来源**:
- SincConv & Residual_block: AASIST.py
- WavLM Frontend: MoEMambaASV.py
- Pre-Norm BiMamba: Fake-Mamba-main
- 融合模块: 必要适配（原始代码库无双流融合）

**文档**: `models/DualStreamSEMamba/CODE_SOURCE.md`

---

## 📝 文件命名规范

### 模型文件
- 格式: `{ModelName}.py`
- 位置: `models/`
- 示例: `MoEMambaASV.py`, `DualStreamSEMamba.py`

### 配置文件
- 格式: `{ModelName}.conf`
- 位置: `config/`
- 示例: `MoEMambaASV.conf`, `DualStreamSEMamba.conf`

### 训练日志
- 格式: `train_{model}_{comment}.log`
- 位置: 根目录或 `exp_result/`
- 示例: `train_phase4.log`, `train_moe_mamba.log`

### 评估结果
- 格式: `eval_{model}_epoch{epoch}_{dataset}.log`
- 位置: 根目录或 `exp_result/`
- 示例: `eval_epoch23_19LA.log`

---

## 🔧 使用说明

### 训练 Phase 3 模型
```bash
python main.py --config ./config/MoEMambaASV.conf
```

### 训练 Phase 4 模型
```bash
python main.py --config ./config/DualStreamSEMamba.conf
```

### 评估模型
```bash
python main.py --eval --config ./config/MoEMambaASV.conf
```

---

## 📊 实验结果组织

```
exp_result/
├── MoE-Mamba-ASV_20251206_182658/    # Phase 3 训练结果
│   ├── LA_MoEMambaASV_ep50_bs16/
│   │   ├── weights/                  # 模型权重
│   │   ├── metrics/                  # 训练指标
│   │   └── eval_scores_*.txt         # 评估分数
│   └── train.log                     # 训练日志
│
└── DualStreamSEMamba_*/              # Phase 4 训练结果（待生成）
    └── ...
```

---

## 🗑️ 已清理文件

以下文件已被删除（过时模型）:
- ❌ `models/CascadeMamba.py`
- ❌ `models/WavLMMamba.py`
- ❌ `models/AASISTMamba.py`
- ❌ `models/Wav2Vec2AASIST.py`
- ❌ `config/CascadeMamba*.conf`
- ❌ `config/WavLMMamba.conf`
- ❌ 10+ 个过时的 Markdown 文档

---

**文档版本**: v1.0  
**最后更新**: 2025-01-XX



