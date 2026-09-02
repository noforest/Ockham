"""Detection metrics read back from the per-sample JSONL; pAcc is the primary one."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_results(path):
    return pd.read_json(path, lines=True)


def _pairwise(df):
    """P-C/P-V/P-B/P-R by pair_id; a pair holding an unreadable reply is dropped."""
    counts = {"P-C": 0, "P-V": 0, "P-B": 0, "P-R": 0}
    n_pairs = 0
    n_dropped = 0
    outcome = {(1, 0): "P-C", (1, 1): "P-V", (0, 0): "P-B", (0, 1): "P-R"}
    for _, group in df.groupby("pair_id"):
        vuln = group[group.label == 1]
        benign = group[group.label == 0]
        if len(vuln) != 1 or len(benign) != 1:
            continue
        key = outcome.get((vuln.iloc[0].prediction, benign.iloc[0].prediction))
        if key is None:
            n_dropped += 1
            continue
        n_pairs += 1
        counts[key] += 1
    return counts, n_pairs, n_dropped


def _calibration_error(y_true, p, n_bins=10):
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(float) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _prob_metrics(df_parsed):
    """AUPRC / AUROC / Brier / calibration on rows carrying a probability and both classes."""
    prob = df_parsed[df_parsed.p_vulnerable.notna()]
    y = prob.label.to_numpy()
    p = prob.p_vulnerable.to_numpy()
    nan = float("nan")
    if len(prob) == 0 or len(set(y.tolist())) < 2:
        return {"AUPRC": nan, "AUROC": nan, "Brier": nan, "calibration_error": nan,
                "n_prob_samples": int(len(prob))}
    return {
        "AUPRC": float(average_precision_score(y, p)),
        "AUROC": float(roc_auc_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "calibration_error": _calibration_error(y, p),
        "n_prob_samples": int(len(prob)),
    }


def compute_metrics(df):
    nan = float("nan")
    clean = df[df.n_backend_failures == 0] if len(df) else df
    parsed = clean[clean.prediction != -1]
    has = len(parsed) > 0
    y_true, y_pred = (parsed.label, parsed.prediction) if has else ([], [])

    counts, n_pairs, n_pairs_dropped = _pairwise(clean)
    out = {
        "n_samples": len(df),
        "n_samples_excluding_backend_failures": len(clean),
        "n_pairs": n_pairs,
        "n_pairs_dropped_unparsable": n_pairs_dropped,
        "pAcc": counts["P-C"] / n_pairs if n_pairs else nan,
        **counts,
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if has else nan,
        "F1": float(f1_score(y_true, y_pred)) if has else nan,
        "F1_trivial": float(f1_score(y_true, [1] * len(y_true))) if has else nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if has else nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if has else nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if has else nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if has else nan,
        "unparsable_rate": 1 - len(parsed) / len(clean) if len(clean) else nan,
    }
    out.update(_prob_metrics(parsed) if has else
               {"AUPRC": nan, "AUROC": nan, "Brier": nan, "calibration_error": nan,
                "n_prob_samples": 0})
    out.update({
        "mean_pack_tokens": float(df.pack_tokens.mean()) if len(df) else nan,
        "mean_evidence_tokens": float(df.n_evidence_tokens.mean()) if len(df) else nan,
        "mean_candidates_pool": float(df.n_candidates_pool.mean()) if len(df) else nan,
        "mean_candidates_selected": float(df.n_candidates_selected.mean()) if len(df) else nan,
        "median_index_time_s": float(df.index_time_s.median()) if len(df) else nan,
        "median_build_cold_ms": float(df.build_time_cold_ms.median()) if len(df) else nan,
        "median_build_warm_ms": float(df.build_time_warm_ms.median()) if len(df) else nan,
        "n_backend_failures_rate": float((df.n_backend_failures > 0).mean()) if len(df) else nan,
    })
    return out


def cell_key(df):
    """The (selector, representation, budget, backend) identity a results file carries."""
    first = df.iloc[0]
    return (first.get("selector"), first.get("representation"),
            int(first.get("budget", 0)), first.get("backend", "ts"))


def load_cells(results_dir, pattern="results_*.jsonl"):
    """Every results file in a directory as {cell_id: (metrics, df, path)}."""
    cells = {}
    for path in sorted(Path(results_dir).glob(pattern)):
        try:
            df = load_results(path)
        except ValueError:
            continue                      # empty file, e.g. an interrupted run
        if not len(df):
            continue
        sel, rep, budget, arm = cell_key(df)
        cells[f"{sel}_{rep}_b{budget}_{arm}"] = (compute_metrics(df), df, path)
    return cells


if __name__ == "__main__":
    path = Path(sys.argv[1])
    metrics = compute_metrics(load_results(path))
    for k, v in metrics.items():
        print(f"{k}: {v}")
    out_path = path.with_name(path.name.replace("results_", "metrics_", 1)).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"-> {out_path}")
