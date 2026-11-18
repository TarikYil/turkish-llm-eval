# -*- coding: utf-8 -*-
"""
Recompute the paper's statistics from the released per-item results (no GPU, no model).

  python scripts/recompute_stats.py              # print summary
  python scripts/recompute_stats.py --csv out/   # also write scores.csv and mcnemar.csv

Statistics are identical to the notebook (stats cell):
  * Wilson score interval (95%) for accuracy / EM / MC1 / hallucination accuracy
  * percentile bootstrap (10,000 resamples, seed 42) for TruthfulQA MC2
  * paired McNemar test: exact binomial if b + c <= 25, otherwise chi-square with
    Edwards continuity correction; significance threshold p < 0.05
Requires: numpy, scipy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

BENCHES = ["mmlu", "arc", "hellaswag", "gsm8k", "truthfulqa", "hallucination"]
PRIMARY = {"mmlu": "accuracy", "arc": "accuracy", "hellaswag": "accuracy",
           "gsm8k": "exact_match", "truthfulqa": "mc1", "hallucination": "accuracy"}


def wilson_ci(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(x, conf: float = 0.95, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - conf) / 2
    return float(x.mean()), float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def mcnemar_p(a, b, exact_threshold: int = 25) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    assert len(a) == len(b), "models must be evaluated on the same items"
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n = n10 + n01
    if n == 0:
        return 1.0
    if n <= exact_threshold:
        return min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5))
    return float(stats.chi2.sf((abs(n10 - n01) - 1) ** 2 / n, df=1))


def arc_digit_key_indices(manifest: Path) -> list[int]:
    with manifest.open(encoding="utf-8") as f:
        return [int(r["eval_index"]) for r in csv.DictReader(f)
                if r["benchmark"] == "arc" and r["flag"] == "arc_digit_key" and r["evaluated"] == "1"]


def apply_arc_correction(models: dict, manifest: Path):
    """Known issue: ARC digit answer keys are 1-based but were read as 0-based.
    Re-score those items from the stored predictions (pred == gold - 1)."""
    idx = arc_digit_key_indices(manifest)
    for key, d in models.items():
        r = d["benchmarks"].get("arc")
        if r is None:
            continue
        if not r.get("pred"):
            print(f"  [arc-corrected] {key}: no stored predictions, ARC removed from this run")
            del d["benchmarks"]["arc"]
            continue
        ic = list(r["item_correct"])
        for i in idx:
            ic[i] = int(int(r["pred"][i]) == int(r["gold"][i]) - 1)
        r["item_correct"] = ic


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--arc-corrected", action="store_true",
                    help="re-score the ARC items with 1-based digit keys (see README, Known issue)")
    args = ap.parse_args()

    models = {}
    for p in sorted(args.results.glob("result_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        models[d["model_key"]] = d
    if not models:
        raise SystemExit(f"no result_*.json in {args.results}")
    if args.arc_corrected:
        apply_arc_correction(models, root / "data" / "subset_manifest.csv")

    score_rows, pair_rows = [], []
    for bench in BENCHES:
        print(f"\n### {bench}  ({PRIMARY[bench]}, 95% Wilson CI)")
        for key, d in models.items():
            r = d["benchmarks"].get(bench)
            if r is None:
                continue
            ic = r["item_correct"]
            p, lo, hi = wilson_ci(int(sum(ic)), len(ic))
            extra = ""
            row = {"model": key, "benchmark": bench, "metric": PRIMARY[bench],
                   "n": len(ic), "score": round(p * 100, 1),
                   "ci_low": round(lo * 100, 1), "ci_high": round(hi * 100, 1)}
            if bench == "truthfulqa" and r.get("mc2_per_item"):
                m, mlo, mhi = bootstrap_ci(r["mc2_per_item"])
                extra = f"  | MC2 {m*100:.1f} [{mlo*100:.1f}, {mhi*100:.1f}]"
                row.update(mc2=round(m * 100, 1), mc2_low=round(mlo * 100, 1),
                           mc2_high=round(mhi * 100, 1))
            if bench == "hallucination":
                extra = (f"  | F1 {r['f1']*100:.1f} P {r['precision']*100:.1f} "
                         f"R {r['recall']*100:.1f} (pos={r['n_pos']}, neg={r['n_neg']})")
                row.update(f1=round(r["f1"] * 100, 1))
            if "macro_f1" in r:
                row.update(macro_f1=round(r["macro_f1"] * 100, 1))
            score_rows.append(row)
            print(f"  {key:<22} n={len(ic):<5} {p*100:5.1f} [{lo*100:.1f}, {hi*100:.1f}]{extra}")

        keys = [k for k in models if bench in models[k]["benchmarks"]]
        sig = 0
        for a, b in combinations(keys, 2):
            ia = models[a]["benchmarks"][bench]["item_correct"]
            ib = models[b]["benchmarks"][bench]["item_correct"]
            pv = mcnemar_p(ia, ib)
            sig += pv < 0.05
            pair_rows.append({"benchmark": bench, "model_a": a, "model_b": b,
                              "diff_pp": round((np.mean(ia) - np.mean(ib)) * 100, 1),
                              "p_value": pv, "significant_05": pv < 0.05})
        if len(keys) >= 2:
            print(f"  McNemar p<0.05: {sig}/{len(keys)*(len(keys)-1)//2} pairs")

    if args.csv:
        args.csv.mkdir(parents=True, exist_ok=True)
        for name, rows in (("scores.csv", score_rows), ("mcnemar.csv", pair_rows)):
            cols = sorted({c for r in rows for c in r}, key=lambda c: list(rows[0]).index(c)
                          if c in rows[0] else 99)
            with (args.csv / name).open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
