#!/usr/bin/env python3
"""
检查ASVspoof2021数据集文件的完整性
"""

import os
import sys
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def check_flac_file(file_path):
    """检查单个flac文件是否可以正常读取"""
    try:
        data, sr = sf.read(str(file_path))
        if len(data) == 0:
            return False, "文件为空"
        if sr != 16000:
            return False, f"采样率不正确: {sr} (期望16000)"
        return True, None
    except Exception as e:
        return False, str(e)

def check_dataset(dataset_path, protocol_file=None):
    """检查整个数据集"""
    dataset_path = Path(dataset_path)
    flac_dir = dataset_path / "ASVspoof2021_DF_eval" / "flac"
    
    if not flac_dir.exists():
        print(f"错误: 找不到flac目录: {flac_dir}")
        return
    
    # 获取要检查的文件列表
    if protocol_file and Path(protocol_file).exists():
        print(f"从协议文件读取文件列表: {protocol_file}")
        with open(protocol_file, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
        print(f"协议文件中包含 {len(file_list)} 个文件")
    else:
        print("扫描flac目录中的所有文件...")
        file_list = [f.stem for f in flac_dir.glob("*.flac")]
        print(f"找到 {len(file_list)} 个flac文件")
    
    # 检查文件
    corrupted_files = []
    missing_files = []
    total_size = 0
    valid_count = 0
    
    print("\n开始检查文件...")
    for file_id in tqdm(file_list, desc="检查进度"):
        flac_path = flac_dir / f"{file_id}.flac"
        
        if not flac_path.exists():
            missing_files.append(file_id)
            continue
        
        # 检查文件大小
        file_size = flac_path.stat().st_size
        total_size += file_size
        
        # 检查文件是否可以读取
        is_valid, error_msg = check_flac_file(flac_path)
        if is_valid:
            valid_count += 1
        else:
            corrupted_files.append((file_id, error_msg, file_size))
    
    # 输出统计信息
    print("\n" + "="*60)
    print("数据集检查结果")
    print("="*60)
    print(f"总文件数: {len(file_list)}")
    print(f"有效文件数: {valid_count}")
    print(f"损坏文件数: {len(corrupted_files)}")
    print(f"缺失文件数: {len(missing_files)}")
    print(f"总大小: {total_size / (1024**3):.2f} GB")
    
    if corrupted_files:
        print(f"\n损坏的文件列表 (前50个):")
        print("-" * 60)
        for i, (file_id, error, size) in enumerate(corrupted_files[:50]):
            print(f"{i+1}. {file_id}: {error} (大小: {size} bytes)")
        if len(corrupted_files) > 50:
            print(f"... 还有 {len(corrupted_files) - 50} 个损坏文件")
        
        # 保存损坏文件列表
        corrupted_list_file = dataset_path / "corrupted_files.txt"
        with open(corrupted_list_file, 'w') as f:
            for file_id, error, size in corrupted_files:
                f.write(f"{file_id}\t{error}\t{size}\n")
        print(f"\n所有损坏文件列表已保存到: {corrupted_list_file}")
    
    if missing_files:
        print(f"\n缺失的文件列表 (前50个):")
        print("-" * 60)
        for i, file_id in enumerate(missing_files[:50]):
            print(f"{i+1}. {file_id}")
        if len(missing_files) > 50:
            print(f"... 还有 {len(missing_files) - 50} 个缺失文件")
        
        # 保存缺失文件列表
        missing_list_file = dataset_path / "missing_files.txt"
        with open(missing_list_file, 'w') as f:
            for file_id in missing_files:
                f.write(f"{file_id}\n")
        print(f"\n所有缺失文件列表已保存到: {missing_list_file}")
    
    # 检查文件大小分布
    if corrupted_files:
        sizes = [size for _, _, size in corrupted_files]
        print(f"\n损坏文件大小统计:")
        print(f"  最小: {min(sizes)} bytes")
        print(f"  最大: {max(sizes)} bytes")
        print(f"  平均: {sum(sizes)/len(sizes):.2f} bytes")
    
    print("="*60)
    
    return corrupted_files, missing_files

if __name__ == "__main__":
    dataset_path = "/root/autodl-tmp/ASVspoof2021_DF"
    protocol_file = "/root/autodl-tmp/ASVspoof2021_DF/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt"
    
    print("ASVspoof2021数据集完整性检查")
    print("="*60)
    print(f"数据集路径: {dataset_path}")
    print(f"协议文件: {protocol_file}")
    print("="*60 + "\n")
    
    corrupted, missing = check_dataset(dataset_path, protocol_file)
    
    if corrupted or missing:
        print(f"\n⚠️  发现 {len(corrupted)} 个损坏文件和 {len(missing)} 个缺失文件")
    else:
        print("\n✅ 所有文件检查通过！")

