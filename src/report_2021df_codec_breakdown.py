import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_eer_minflip(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """
    Compute EER (%), taking min of score and -score to be robust to sign conventions.
    """
    if bonafide_scores.size == 0 or spoof_scores.size == 0:
        return float("nan")

    def _eer(y_true, y_score):
        # manual EER without sklearn to keep deps minimal
        scores = np.asarray(y_score, dtype=np.float64)
        labels = np.asarray(y_true, dtype=np.int64)
        idx = np.argsort(scores, kind="mergesort")
        labels = labels[idx]
        # target=bonafide=1, non-target=spoof=0
        tar = labels.sum()
        non = labels.size - tar
        tar_cum = np.cumsum(labels)
        non_cum = non - (np.arange(1, labels.size + 1) - tar_cum)
        frr = np.concatenate(([0.0], tar_cum / max(tar, 1)))
        far = np.concatenate(([1.0], non_cum / max(non, 1)))
        k = np.argmin(np.abs(frr - far))
        return float(100.0 * 0.5 * (frr[k] + far[k]))

    y_true = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    y_score = np.concatenate([bonafide_scores, spoof_scores])
    eer1 = _eer(y_true, y_score)
    eer2 = _eer(y_true, -y_score)
    return min(eer1, eer2)


@dataclass
class MetaRow:
    codec: str
    source: str
    key: str  # bonafide/spoof


def load_scores(score_file: Path) -> Dict[str, float]:
    scores = {}
    with score_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt = parts[0]
            try:
                score = float(parts[-1])
            except Exception:
                continue
            scores[utt] = score
    return scores


def parse_key_line(parts: List[str]) -> Tuple[str, MetaRow]:
    # trial_metadata.txt: SPK FILE CODEC SRC ATTACK_ID KEY ...
    # Example:
    # LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof ...
    fname = parts[1]
    codec = parts[2]
    source = parts[3]
    key = parts[5]
    return fname, MetaRow(codec=codec, source=source, key=key)


def main():
    ap = argparse.ArgumentParser(description="ASVspoof2021 DF EER report with codec breakdown (paper-friendly).")
    ap.add_argument("--score_file", required=True, help="Score file with 'utt_id score' or 'utt_id ... score'")
    ap.add_argument("--key_file", required=True, help="keys/DF/CM/trial_metadata.txt")
    ap.add_argument("--out", default="report_2021df_codec.md", help="Output markdown path")
    args = ap.parse_args()

    score_file = Path(args.score_file)
    key_file = Path(args.key_file)
    out_file = Path(args.out)

    scores = load_scores(score_file)

    # accumulate scores per group
    bonafide_all = []
    spoof_all = []
    by_codec_bona = defaultdict(list)
    by_codec_spoof = defaultdict(list)
    by_source_bona = defaultdict(list)
    by_source_spoof = defaultdict(list)

    with key_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            utt, meta = parse_key_line(parts)
            if utt not in scores:
                continue
            s = scores[utt]
            if meta.key == "bonafide":
                bonafide_all.append(s)
                by_codec_bona[meta.codec].append(s)
                by_source_bona[meta.source].append(s)
            else:
                spoof_all.append(s)
                by_codec_spoof[meta.codec].append(s)
                by_source_spoof[meta.source].append(s)

    bonafide_all = np.array(bonafide_all, dtype=np.float64)
    spoof_all = np.array(spoof_all, dtype=np.float64)
    overall_eer = compute_eer_minflip(bonafide_all, spoof_all)

    lines = []
    lines.append(f"# ASVspoof 2021 DF Report (Codec Breakdown)\n")
    lines.append(f"- **Score file**: `{score_file}`")
    lines.append(f"- **Key file**: `{key_file}`")
    lines.append(f"- **Total bonafide**: {bonafide_all.size}")
    lines.append(f"- **Total spoof**: {spoof_all.size}")
    lines.append(f"- **Overall EER (minflip)**: **{overall_eer:.3f}%**\n")

    # Codec breakdown (within-codec bonafide vs within-codec spoof)
    lines.append("## Breakdown by Codec\n")
    lines.append("| Codec | EER (%) | Bonafide | Spoof | Total |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for codec in sorted(set(list(by_codec_bona.keys()) + list(by_codec_spoof.keys()))):
        b = np.array(by_codec_bona.get(codec, []), dtype=np.float64)
        s = np.array(by_codec_spoof.get(codec, []), dtype=np.float64)
        eer = compute_eer_minflip(b, s) if (b.size > 0 and s.size > 0) else float("nan")
        lines.append(f"| {codec} | {eer:.3f} | {b.size} | {s.size} | {b.size + s.size} |")

    # Source breakdown
    lines.append("\n## Breakdown by Source Domain\n")
    lines.append("| Source | EER (%) | Bonafide | Spoof | Total |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for src in sorted(set(list(by_source_bona.keys()) + list(by_source_spoof.keys()))):
        b = np.array(by_source_bona.get(src, []), dtype=np.float64)
        s = np.array(by_source_spoof.get(src, []), dtype=np.float64)
        eer = compute_eer_minflip(b, s) if (b.size > 0 and s.size > 0) else float("nan")
        lines.append(f"| {src} | {eer:.3f} | {b.size} | {s.size} | {b.size + s.size} |")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote -> {out_file}")


if __name__ == "__main__":
    main()


