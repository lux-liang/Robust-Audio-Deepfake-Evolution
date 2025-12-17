# 磁盘清理报告

## 📊 清理前状态
- **磁盘使用率**: 80% (24G/30G)
- **剩余空间**: 6.2GB
- **状态**: ⚠️ 空间不足，训练可能中断

## ✅ 已执行的清理操作

### 1. 删除压缩包文件（约2.2GB）
- ✅ `SOTA_CascadeMamba_EER0.274.zip` (1.4GB)
- ✅ `SOTA_Backup_0.551_Epoch18.zip` (714MB)
- ✅ `RawBMamba-main.zip` (20MB)
- ✅ `tDCF_python_v2.zip` (8.2MB)
- ✅ `AASIST_Reproduction_Report.zip` (4.7KB)

### 2. 清理 __pycache__ 目录
- ✅ 已清理所有 `__pycache__` 目录（约256KB）

### 3. 清理旧的实验结果
- ✅ 清理 `exp_result/MoE-Mamba-ASV_20251206_182658` 中的中间checkpoint
  - 保留: `best.pth`
  - 删除: 所有 `checkpoint_*.pth` 和中间 `epoch_*.pth`
- ✅ 删除 `exp_result/WavLM-Mamba_20251206_180420` (1.2GB) - Phase 2 旧实验

## 📊 清理后状态
- **磁盘使用率**: 73% (22G/30G)
- **剩余空间**: 8.3GB
- **释放空间**: 约 27GB (删除旧实验目录)
- **状态**: ✅ 空间充足，可以开始训练

## ✅ 已删除的大目录
- ✅ `exp_result/cascade_mamba` (15GB) - Phase 1-2 旧实验
- ✅ `exp_result/cascade_mamba_supcon_finetune` (9.5GB) - 旧实验

## 🔍 可进一步清理的大目录

### 高优先级（可释放大量空间）

1. **`exp_result/cascade_mamba` (15GB)**
   - Phase 1-2 的旧实验
   - 建议: 如果不需要，可以删除整个目录
   - 命令: `rm -rf exp_result/cascade_mamba`

2. **`exp_result/cascade_mamba_supcon_finetune` (9.5GB)**
   - 旧实验
   - 建议: 如果不需要，可以删除
   - 命令: `rm -rf exp_result/cascade_mamba_supcon_finetune`

3. **`exp_result/MoE-Mamba-ASV_20251206_182658` (27GB)**
   - Phase 3 的实验结果
   - 建议: 保留最佳模型，删除其他文件
   - 已清理中间checkpoint，但可能还有其他大文件（日志、tensorboard等）

### 中优先级

4. **`backup_models/` (2.4GB)**
   - 包含两个旧模型备份
   - 建议: 如果已备份到本地，可以删除
   - 命令: `rm -rf backup_models/*`

## 🛠️ 已优化的保存策略

### 1. 最佳模型保存优化
- **修改前**: 每次找到更好的模型都保存一个新文件，导致多个文件累积
- **修改后**: 只保留最新的最佳模型，自动删除旧的
- **位置**: `main.py` 第 269-275 行

### 2. Checkpoint 保存优化
- **修改前**: 每10个epoch保存一次，不删除旧的
- **修改后**: 只保留最近3个checkpoint，自动删除更早的
- **位置**: `main.py` 第 303-315 行

### 3. 保存策略总结
- ✅ 只保存最佳模型（自动删除旧的）
- ✅ 只保留最近3个checkpoint
- ✅ 每10个epoch保存一次checkpoint（可配置）

## 📋 训练前建议

### 必须执行（如果空间仍然不足）

```bash
# 删除 Phase 1-2 的旧实验（如果不需要）
rm -rf exp_result/cascade_mamba
rm -rf exp_result/cascade_mamba_supcon_finetune

# 如果已备份到本地，删除备份模型
rm -rf backup_models/*
```

### 可选执行

```bash
# 清理 MoE-Mamba 实验结果中的其他大文件（日志、tensorboard等）
# 注意: 只删除不需要的文件，保留最佳模型
find exp_result/MoE-Mamba-ASV_20251206_182658 -name "*.log" -delete
find exp_result/MoE-Mamba-ASV_20251206_182658 -name "events.out.tfevents.*" -delete
```

## ⚠️ 训练时注意事项

1. **监控磁盘空间**: 训练过程中定期检查 `df -h`
2. **模型大小**: 每个模型权重文件约 1.2GB
3. **Checkpoint**: 已优化为只保留最近3个
4. **最佳模型**: 已优化为只保留最新的一个

## 📊 预期空间需求

- **每个模型权重**: ~1.2GB
- **最佳模型**: 1个 = 1.2GB
- **Checkpoint**: 3个 = 3.6GB
- **训练日志**: ~100MB
- **总计**: 约 5GB

**建议**: 至少保留 10GB 可用空间，以确保训练不会因空间不足而中断。

---

**清理日期**: 2025-01-XX  
**清理脚本**: `cleanup_disk.sh`  
**状态**: ✅ 部分清理完成，建议进一步清理大目录

