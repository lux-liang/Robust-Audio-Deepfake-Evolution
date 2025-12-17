import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# 配置
SCORE_FILE = "/root/aasist-main/exp_result/cascade_mamba_2021DF_eval/eval_scores_2021DF.txt"
# 使用相对路径或绝对路径，之前解压在当前目录下的 keys
KEY_FILE = "keys/DF/CM/trial_metadata.txt"

def compute_eer(bonafide_scores, spoof_scores):
    if len(bonafide_scores) < 1 or len(spoof_scores) < 1:
        return np.nan
    
    y_true = [1] * len(bonafide_scores) + [0] * len(spoof_scores)
    y_scores = np.concatenate((bonafide_scores, spoof_scores))
    
    # 反转分数检测 (Phase 2.5 模型在 2021 DF 上通常需要反转)
    # 这里我们先假设需要反转（因为之前的计算表明反转后 EER 更低）
    # 为了保险，我们计算两种情况取最小值
    
    # Case 1: Normal
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer_1 = fpr[idx] * 100
    
    # Case 2: Inverted
    fpr, tpr, _ = roc_curve(y_true, -y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer_2 = fpr[idx] * 100
    
    return min(eer_1, eer_2)

def main():
    print("🚀 Loading data for detailed breakdown analysis...")
    
    # 1. Load Scores
    scores = {}
    with open(SCORE_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scores[parts[0]] = float(parts[1])
                
    # 2. Load Metadata & Organize
    data = []
    # Metadata format: SPK FILE CODEC SRC ATTACK_ID KEY ...
    with open(KEY_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 5:
                fname = parts[1]
                if fname not in scores: continue
                
                codec = parts[2]
                src = parts[3]     # asvspoof, vcc2020, vcc2018
                key = parts[5]     # bonafide, spoof
                
                # Enhance details
                vocoder = parts[7] if len(parts) > 7 else "unknown"
                
                data.append({
                    'file': fname,
                    'score': scores[fname],
                    'key': key,
                    'codec': codec,
                    'source': src,
                    'vocoder': vocoder
                })
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} samples.")
    
    # 3. Breakdown Analysis
    
    # A. By Codec (Compression Type)
    print("\n📊 Breakdown by Codec (Compression):")
    print("-" * 60)
    print(f"{'Codec':<15} | {'EER (%)':<10} | {'Count':<8}")
    print("-" * 60)
    
    codecs = df['codec'].unique()
    bonafide_all = df[df['key'] == 'bonafide']['score'].values
    
    for c in sorted(codecs):
        subset = df[df['codec'] == c]
        spoof_sub = subset[subset['key'] == 'spoof']['score'].values
        # Note: Bonafide samples also have codecs, so we should compare 
        # bonafide(codec_X) vs spoof(codec_X) if possible, 
        # OR bonafide(all) vs spoof(codec_X) depending on protocol.
        # ASVspoof 2021 DF protocol usually compares all bonafide vs specific spoof subsets
        # but strictly speaking, bonafide samples are also compressed.
        
        # Let's use subset bonafide vs subset spoof to be precise about codec impact
        bonafide_sub = subset[subset['key'] == 'bonafide']['score'].values
        
        if len(spoof_sub) > 0 and len(bonafide_sub) > 0:
            eer = compute_eer(bonafide_sub, spoof_sub)
            print(f"{c:<15} | {eer:<10.2f} | {len(subset):<8}")
        else:
            print(f"{c:<15} | {'N/A':<10} | {len(subset):<8}")

    # B. By Source Domain (vcc vs asvspoof)
    print("\n📊 Breakdown by Source Domain:")
    print("-" * 60)
    print(f"{'Source':<15} | {'EER (%)':<10} | {'Count':<8}")
    print("-" * 60)
    
    sources = df['source'].unique()
    for s in sorted(sources):
        subset = df[df['source'] == s]
        bonafide_sub = subset[subset['key'] == 'bonafide']['score'].values
        spoof_sub = subset[subset['key'] == 'spoof']['score'].values
        
        if len(spoof_sub) > 0 and len(bonafide_sub) > 0:
            eer = compute_eer(bonafide_sub, spoof_sub)
            print(f"{s:<15} | {eer:<10.2f} | {len(subset):<8}")
        else:
            # Sometimes bonafide is only labeled as 'vcc2020' or similar? 
            # Let's check global bonafide if subset is empty
             eer = compute_eer(bonafide_all, spoof_sub)
             print(f"{s:<15} | {eer:<10.2f}*| {len(subset):<8}")

    # C. By Vocoder Type (for spoof only)
    print("\n📊 Breakdown by Vocoder (Spoof Only vs All Bonafide):")
    print("-" * 60)
    print(f"{'Vocoder':<30} | {'EER (%)':<10} | {'Count':<8}")
    print("-" * 60)
    
    vocoders = df[df['key'] == 'spoof']['vocoder'].unique()
    for v in sorted(vocoders):
        if v == '-' or v == 'unknown': continue
        spoof_sub = df[(df['key'] == 'spoof') & (df['vocoder'] == v)]['score'].values
        
        eer = compute_eer(bonafide_all, spoof_sub)
        print(f"{v:<30} | {eer:<10.2f} | {len(spoof_sub):<8}")

if __name__ == "__main__":
    main()




import pandas as pd
from sklearn.metrics import roc_curve

# 配置
SCORE_FILE = "/root/aasist-main/exp_result/cascade_mamba_2021DF_eval/eval_scores_2021DF.txt"
# 使用相对路径或绝对路径，之前解压在当前目录下的 keys
KEY_FILE = "keys/DF/CM/trial_metadata.txt"

def compute_eer(bonafide_scores, spoof_scores):
    if len(bonafide_scores) < 1 or len(spoof_scores) < 1:
        return np.nan
    
    y_true = [1] * len(bonafide_scores) + [0] * len(spoof_scores)
    y_scores = np.concatenate((bonafide_scores, spoof_scores))
    
    # 反转分数检测 (Phase 2.5 模型在 2021 DF 上通常需要反转)
    # 这里我们先假设需要反转（因为之前的计算表明反转后 EER 更低）
    # 为了保险，我们计算两种情况取最小值
    
    # Case 1: Normal
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer_1 = fpr[idx] * 100
    
    # Case 2: Inverted
    fpr, tpr, _ = roc_curve(y_true, -y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer_2 = fpr[idx] * 100
    
    return min(eer_1, eer_2)

def main():
    print("🚀 Loading data for detailed breakdown analysis...")
    
    # 1. Load Scores
    scores = {}
    with open(SCORE_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scores[parts[0]] = float(parts[1])
                
    # 2. Load Metadata & Organize
    data = []
    # Metadata format: SPK FILE CODEC SRC ATTACK_ID KEY ...
    with open(KEY_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 5:
                fname = parts[1]
                if fname not in scores: continue
                
                codec = parts[2]
                src = parts[3]     # asvspoof, vcc2020, vcc2018
                key = parts[5]     # bonafide, spoof
                
                # Enhance details
                vocoder = parts[7] if len(parts) > 7 else "unknown"
                
                data.append({
                    'file': fname,
                    'score': scores[fname],
                    'key': key,
                    'codec': codec,
                    'source': src,
                    'vocoder': vocoder
                })
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} samples.")
    
    # 3. Breakdown Analysis
    
    # A. By Codec (Compression Type)
    print("\n📊 Breakdown by Codec (Compression):")
    print("-" * 60)
    print(f"{'Codec':<15} | {'EER (%)':<10} | {'Count':<8}")
    print("-" * 60)
    
    codecs = df['codec'].unique()
    bonafide_all = df[df['key'] == 'bonafide']['score'].values
    
    for c in sorted(codecs):
        subset = df[df['codec'] == c]
        spoof_sub = subset[subset['key'] == 'spoof']['score'].values
        # Note: Bonafide samples also have codecs, so we should compare 
        # bonafide(codec_X) vs spoof(codec_X) if possible, 
        # OR bonafide(all) vs spoof(codec_X) depending on protocol.
        # ASVspoof 2021 DF protocol usually compares all bonafide vs specific spoof subsets
        # but strictly speaking, bonafide samples are also compressed.
        
        # Let's use subset bonafide vs subset spoof to be precise about codec impact
        bonafide_sub = subset[subset['key'] == 'bonafide']['score'].values
        
        if len(spoof_sub) > 0 and len(bonafide_sub) > 0:
            eer = compute_eer(bonafide_sub, spoof_sub)
            print(f"{c:<15} | {eer:<10.2f} | {len(subset):<8}")
        else:
            print(f"{c:<15} | {'N/A':<10} | {len(subset):<8}")

    # B. By Source Domain (vcc vs asvspoof)
    print("\n📊 Breakdown by Source Domain:")
    print("-" * 60)
    print(f"{'Source':<15} | {'EER (%)':<10} | {'Count':<8}")
    print("-" * 60)
    
    sources = df['source'].unique()
    for s in sorted(sources):
        subset = df[df['source'] == s]
        bonafide_sub = subset[subset['key'] == 'bonafide']['score'].values
        spoof_sub = subset[subset['key'] == 'spoof']['score'].values
        
        if len(spoof_sub) > 0 and len(bonafide_sub) > 0:
            eer = compute_eer(bonafide_sub, spoof_sub)
            print(f"{s:<15} | {eer:<10.2f} | {len(subset):<8}")
        else:
            # Sometimes bonafide is only labeled as 'vcc2020' or similar? 
            # Let's check global bonafide if subset is empty
             eer = compute_eer(bonafide_all, spoof_sub)
             print(f"{s:<15} | {eer:<10.2f}*| {len(subset):<8}")

    # C. By Vocoder Type (for spoof only)
    print("\n📊 Breakdown by Vocoder (Spoof Only vs All Bonafide):")
    print("-" * 60)
    print(f"{'Vocoder':<30} | {'EER (%)':<10} | {'Count':<8}")
    print("-" * 60)
    
    vocoders = df[df['key'] == 'spoof']['vocoder'].unique()
    for v in sorted(vocoders):
        if v == '-' or v == 'unknown': continue
        spoof_sub = df[(df['key'] == 'spoof') & (df['vocoder'] == v)]['score'].values
        
        eer = compute_eer(bonafide_all, spoof_sub)
        print(f"{v:<30} | {eer:<10.2f} | {len(spoof_sub):<8}")

if __name__ == "__main__":
    main()

