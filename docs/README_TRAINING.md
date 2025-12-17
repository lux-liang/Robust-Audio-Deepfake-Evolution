# Phase 4 训练指南

## 🚀 快速开始

### 启动训练

```bash
# 方式1: 使用训练启动脚本（推荐）
./start_training_phase4.sh

# 方式2: 直接运行（后台）
nohup python main.py --config ./config/DualStreamSEMamba.conf > train_phase4.log 2>&1 &
```

### 监控训练

```bash
# 使用监控脚本
./monitor_training.sh

# 或手动查看日志
tail -f train_phase4_*.log

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看磁盘空间
watch -n 60 df -h /
```

## 📋 训练前检查清单

- [x] 磁盘空间充足（当前: 8.3GB 可用）
- [x] 配置文件正确 (`config/DualStreamSEMamba.conf`)
- [x] 模型代码完整 (`models/DualStreamSEMamba.py`)
- [x] 数据路径正确
- [x] GPU 可用
- [x] 所有依赖已安装

## ⚙️ 训练配置

### 当前配置 (`DualStreamSEMamba.conf`)

- **模型**: Dual-Stream SE-Mamba
- **Batch Size**: 12
- **Epochs**: 50
- **Loss**: CrossEntropy (初期)
- **Learning Rate**: 5e-5 (Backbone), 1e-6 (WavLM)
- **数据增强**: RawBoost (algo=5), Codec Augmentation
- **保存策略**: 
  - 只保留最新最佳模型
  - 只保留最近3个checkpoint

## 📊 预期训练时间

- **每个 Epoch**: 约 15-30 分钟（取决于GPU）
- **总训练时间**: 约 12-25 小时（50 epochs）

## 🔍 关键监控指标

### 正常训练指标
- **Loss**: 应该平稳下降
- **Dev EER**: 应该逐步下降
- **GPU 使用率**: 应该 > 80%
- **显存使用**: 应该 < 20GB (RTX 4090D 24GB)

### 异常信号
- ❌ **Loss NaN**: 检查梯度裁剪、学习率
- ❌ **Loss 不下降**: 检查学习率、数据加载
- ❌ **显存溢出**: 减小 batch size
- ❌ **磁盘空间不足**: 检查保存策略

## 🛠️ 常用命令

### 查看训练进度
```bash
# 查看最新日志
tail -f train_phase4_*.log

# 查看最新评估结果
grep "dev_eer" train_phase4_*.log | tail -5

# 查看Loss曲线
grep "Loss:" train_phase4_*.log | tail -10
```

### 停止训练
```bash
# 查找训练进程
ps aux | grep "python main.py"

# 停止训练（替换PID）
kill <PID>
```

### 恢复训练
```bash
# 从checkpoint恢复
python main.py --config ./config/DualStreamSEMamba.conf --resume <checkpoint_path>
```

## 📁 训练输出

训练结果将保存在：
```
exp_result/DualStreamSEMamba_<timestamp>/
├── weights/
│   ├── best.pth              # 最佳模型
│   ├── epoch_XX_X.XXX.pth    # 最佳模型（带EER）
│   └── checkpoint_epoch_XX.pth  # 定期checkpoint
├── metrics/
│   ├── dev_score.txt
│   └── dev_t-DCF_EER_XXepo.txt
└── config.conf               # 训练配置备份
```

## 🎯 训练目标

- **Dev EER**: < 2.0% (Phase 3 是 1.139%)
- **Eval EER**: < 10% (Phase 3 是 9.17%)
- **A19 EER**: < 5% (Phase 3 是 23%，目标是显著改善)

## 📝 训练日志示例

```
Start training epoch000
Loss:0.12345, dev_eer: 5.234, dev_tdcf:0.2345
best model find at epoch 0
Saved best model: epoch_0_5.234.pth
...
```

## ⚠️ 注意事项

1. **不要中断训练**: 使用 `nohup` 或 `screen`/`tmux` 防止断网中断
2. **定期检查磁盘**: 确保有足够空间保存模型
3. **监控GPU温度**: 确保GPU温度正常
4. **保存最佳模型**: 训练完成后及时备份最佳模型

---

**祝训练顺利！期待看到 SOTA 结果！** 🎉



