import os
import sys
import numpy as np
import argparse
from datetime import datetime

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds

def compute_eer_impl(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_eer(bonafide_scores, spoof_scores):
    """
    Returns EER (%)
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0
    
    # Check for NaNs
    if np.isnan(bonafide_scores).any() or np.isnan(spoof_scores).any():
        return 99.9
        
    eer, _ = compute_eer_impl(np.array(bonafide_scores), np.array(spoof_scores))
    return eer * 100


ATTACK_TYPES_2019_LA = {
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
    'A19': 'VC (Waveform)',
    '-': 'Bonafide/Unknown'
}


def read_scores(score_file: str):
    """
    Read score file lines in format: utt_id src key score
    produced by main.py produce_evaluation_file().
    Returns:
      bonafide_scores: list[float]
      spoof_scores_by_src: dict[src -> list[float]]
      all_spoof_scores: list[float]
      total_lines: int
    """
    bonafide_scores = []
    spoof_scores_by_src = {}
    all_spoof = []
    total = 0

    with open(score_file, "r") as f:
        for line in f:
            total += 1
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            utt_id, src, key = parts[0], parts[1], parts[2]
            try:
                score = float(parts[3])
            except Exception:
                continue
            if key == "bonafide":
                bonafide_scores.append(score)
            else:
                spoof_scores_by_src.setdefault(src, []).append(score)
                all_spoof.append(score)
    return bonafide_scores, spoof_scores_by_src, all_spoof, total


def write_markdown_report(output_file: str, model_name: str, score_file: str, bonafide_scores, spoof_scores_by_src, all_spoof_scores):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    keys = sorted(spoof_scores_by_src.keys())
    with open(output_file, "w") as out_f:
        out_f.write(f"# {model_name} Performance Breakdown\n\n")
        out_f.write(f"- **Generated**: {ts}\n")
        out_f.write(f"- **Score file**: `{score_file}`\n")
        out_f.write(f"- **Bonafide count**: {len(bonafide_scores)}\n")
        out_f.write(f"- **Spoof count**: {len(all_spoof_scores)}\n")
        out_f.write(f"- **Attack keys found**: {keys}\n\n")

        out_f.write("## Breakdown by Attack Type (ASVspoof2019 LA)\n\n")
        out_f.write("| Attack ID | Type | EER (%) | Count |\n")
        out_f.write("| :--- | :--- | ---: | ---: |\n")

        b = np.array(bonafide_scores, dtype=np.float64)
        for attack_id in sorted(keys):
            if attack_id == "-":
                continue
            s = np.array(spoof_scores_by_src[attack_id], dtype=np.float64)
            eer = compute_eer(b, s)
            desc = ATTACK_TYPES_2019_LA.get(attack_id, "Unknown")
            out_f.write(f"| {attack_id} | {desc} | **{eer:.3f}** | {len(s)} |\n")

        overall_eer = compute_eer(b, np.array(all_spoof_scores, dtype=np.float64))
        out_f.write(f"\n## Overall\n\n")
        out_f.write(f"- **Overall EER (bonafide vs all spoof)**: **{overall_eer:.3f}%**\n")

def main():
    parser = argparse.ArgumentParser(description="Attack-wise EER breakdown for ASVspoof2019 LA score files.")
    parser.add_argument("--score_file", type=str, required=True, help="Path to score file (utt_id src key score)")
    parser.add_argument("--output_file", type=str, default="analysis_breakdown.md", help="Output markdown report path")
    parser.add_argument("--model_name", type=str, default="Model", help="Name shown in report")
    args = parser.parse_args()

    if not os.path.exists(args.score_file):
        raise FileNotFoundError(f"Score file not found: {args.score_file}")

    bonafide_scores, spoof_scores_by_src, all_spoof, total = read_scores(args.score_file)
    print(f"Read {total} lines from {args.score_file}")
    print(f"Bonafide={len(bonafide_scores)} Spoof={len(all_spoof)} AttackKeys={sorted(spoof_scores_by_src.keys())}")
    write_markdown_report(args.output_file, args.model_name, args.score_file, bonafide_scores, spoof_scores_by_src, all_spoof)
    print(f"Wrote report -> {args.output_file}")

if __name__ == "__main__":
    main()
