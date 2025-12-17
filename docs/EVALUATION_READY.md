# 📊 评估报告状态与 ASV2021 DF 跨库评估准备

**更新时间**: 2025-11-29

---

## ✅ 已完成

### 1. Cascade-Mamba 详细分项报告
- **文件**: `analysis_report_v2.md`
- **总体 EER**: 2.735%
- **包含**: 所有 13 种攻击类型（A07-A19）的详细数据

### 2. ASV2021 DF 跨库评估脚本
- **文件**: `evaluate_2021DF.py`
- **状态**: ✅ 已准备就绪
- **模型**: `epoch_39_0.206.pth` (Phase 2 最佳模型)
- **重要**: 这是**评估（测试）**，不是训练！

---

## 🔄 进行中

### AASIST 基线评估
- **状态**: 等待完成
- **文件**: `exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt`
- **完成后**: 运行 `python compare_models.py` 生成完整对比报告

---

## 🚀 ASV2021 DF 跨库评估操作指南

### ⚠️ 重要提醒
**这是跨库评估（Cross-dataset Evaluation），不是训练！**
- ✅ 模型仅在 **ASVspoof 2019 LA** 上训练
- ✅ 在 **ASVspoof 2021 DF** 上仅进行**推理测试**
- ❌ **绝对不要**在 2021 DF 上训练或微调

### 运行评估

```bash
cd /root/aasist-main
python evaluate_2021DF.py
```

### 预期输出

1. **Score 文件**: `exp_result/cascade_mamba_2021DF_eval/eval_scores_2021DF.txt`
2. **结果报告**: `exp_result/cascade_mamba_2021DF_eval/results_2021DF.txt`
3. **统计信息**: 分数分布、均值、标准差等

### 注意事项

- ASV2021 DF 评估集**不提供标签**，因此无法直接计算 EER
- 生成的 score 文件需要提交到 ASVspoof 2021 官方评估服务器获取官方 EER
- 或者如果有单独的标签文件，可以手动计算 EER

### 参考性能

根据相关论文：
- **Fake-Mamba**: 1.74% EER on 2021 DF
- **BiCrossMamba**: 14.77% EER on 2021 DF
- **目标**: 我们的模型（使用 RawBoost 训练，抗压缩能力强）预期 EER < 5%

---

## 📋 文件清单

### 核心报告
- `analysis_report_v2.md` - Cascade-Mamba 分项报告
- `comparison_report.md` - 对比报告（待 AASIST 完成后更新）

### 评估脚本
- `evaluate_2021DF.py` - ASV2021 DF 跨库评估脚本
- `compare_models.py` - 模型对比脚本
- `analyze_breakdown.py` - 分项分析脚本

### 模型权重
- `exp_result/cascade_mamba/LA_CascadeMamba_ep100_bs16/weights/epoch_39_0.206.pth` - 最佳模型

---

*准备就绪，可以开始 ASV2021 DF 跨库评估！*



