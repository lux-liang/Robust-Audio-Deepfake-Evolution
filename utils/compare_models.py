#!/usr/bin/env python3
"""
Compare AASIST Baseline vs Cascade-Mamba performance breakdown
"""

import os
import sys
import numpy as np

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds

def compute_eer_impl(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_eer(bonafide_scores, spoof_scores):
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0
    if np.isnan(bonafide_scores).any() or np.isnan(spoof_scores).any():
        return 99.9
    eer, _ = compute_eer_impl(np.array(bonafide_scores), np.array(spoof_scores))
    return eer * 100

def analyze_model(score_file, model_name):
    """Analyze a single model's performance"""
    if not os.path.exists(score_file):
        return None
    
    bonafide_scores = []
    spoof_scores_by_id = {}
    
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            src = parts[1]
            key = parts[2]
            
            try:
                score = float(parts[3])
            except (ValueError, IndexError):
                continue
            
            if key == 'bonafide':
                bonafide_scores.append(score)
            else:
                if src not in spoof_scores_by_id:
                    spoof_scores_by_id[src] = []
                spoof_scores_by_id[src].append(score)
    
    bonafide_scores = np.array(bonafide_scores)
    results = {}
    
    for attack_id in sorted(spoof_scores_by_id.keys()):
        if attack_id == '-':
            continue
        scores = np.array(spoof_scores_by_id[attack_id])
        eer = compute_eer(bonafide_scores, scores)
        results[attack_id] = {'eer': eer, 'count': len(scores)}
    
    overall_spoof = []
    for attack_id in spoof_scores_by_id.keys():
        if attack_id != '-':
            overall_spoof.extend(spoof_scores_by_id[attack_id])
    
    overall_eer = compute_eer(bonafide_scores, np.array(overall_spoof))
    results['_overall'] = {'eer': overall_eer, 'count': len(overall_spoof)}
    
    return results

def main():
    # File paths
    baseline_file = "/root/aasist-main/exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt"
    cascade_file = "/root/aasist-main/exp_result/cascade_mamba/LA_CascadeMamba_ep100_bs16/eval_scores_using_best_dev_model.txt"
    
    # Analyze both models
    print("Analyzing Cascade-Mamba...")
    cascade_results = analyze_model(cascade_file, "Cascade-Mamba")
    
    print("Analyzing AASIST Baseline...")
    baseline_results = analyze_model(baseline_file, "AASIST")
    
    if cascade_results is None:
        print("Error: Cascade-Mamba score file not found!")
        return
    
    # Attack type descriptions
    attack_types = {
        'A07': 'TTS (Vocoder)',
        'A08': 'TTS (Vocoder)',
        'A09': 'TTS (Vocoder)',
        'A10': 'TTS (Vocoder)',
        'A11': 'TTS (Vocoder)',
        'A12': 'TTS (Vocoder)',
        'A13': 'TTS-VC',
        'A14': 'TTS-VC',
        'A15': 'TTS-VC',
        'A16': 'TTS (Waveform)',
        'A17': 'VC (Vocoder)',
        'A18': 'VC (Vocoder)',
        'A19': 'VC (Waveform)'
    }
    
    # Generate comparison report
    output_file = "/root/aasist-main/comparison_report.md"
    with open(output_file, 'w') as f:
        f.write("# AASIST Baseline vs Cascade-Mamba Performance Comparison\n\n")
        f.write("## Overall Performance\n\n")
        
        if baseline_results:
            f.write(f"| Model | Overall EER |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **AASIST (Baseline)** | **{baseline_results['_overall']['eer']:.3f}%** |\n")
            f.write(f"| **Cascade-Mamba (Ours)** | **{cascade_results['_overall']['eer']:.3f}%** |\n")
            improvement = ((baseline_results['_overall']['eer'] - cascade_results['_overall']['eer']) / baseline_results['_overall']['eer']) * 100
            f.write(f"\n**Improvement: {improvement:.1f}% reduction in EER**\n\n")
        else:
            f.write("⚠️ **AASIST baseline evaluation not found.**\n")
            f.write("To generate it, run:\n")
            f.write("```bash\n")
            f.write("cd /root/aasist-main\n")
            f.write("python main.py --config ./config/AASIST.conf --eval --eval_model_weights ./exp_result/LA_AASIST_ep100_bs24/weights/epoch_51_0.546.pth --output_dir ./exp_result\n")
            f.write("```\n\n")
            f.write(f"**Cascade-Mamba Overall EER**: {cascade_results['_overall']['eer']:.3f}%\n\n")
        
        f.write("## Detailed Breakdown by Attack Type\n\n")
        f.write("| Attack ID | Type | AASIST EER | Cascade-Mamba EER | Improvement |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for attack_id in sorted([k for k in cascade_results.keys() if k != '_overall']):
            desc = attack_types.get(attack_id, "Unknown")
            cascade_eer = cascade_results[attack_id]['eer']
            
            if baseline_results and attack_id in baseline_results:
                baseline_eer = baseline_results[attack_id]['eer']
                if baseline_eer > 0:
                    improvement = ((baseline_eer - cascade_eer) / baseline_eer) * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"
                f.write(f"| {attack_id} | {desc} | {baseline_eer:.3f}% | **{cascade_eer:.3f}%** | {improvement_str} |\n")
            else:
                f.write(f"| {attack_id} | {desc} | N/A | **{cascade_eer:.3f}%** | - |\n")
    
    print(f"\n✅ Comparison report saved to: {output_file}")
    
    if baseline_results is None:
        print("\n⚠️  Note: AASIST baseline evaluation not found.")
        print("   Run the evaluation command shown above to generate the full comparison.")

if __name__ == "__main__":
    main()




