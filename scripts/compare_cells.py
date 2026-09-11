"""Compare cells -- pAcc, pair breakdown, cost -- with McNemar when a reference is named."""

import json
import sys
from math import comb
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ockham.metrics import compute_metrics

# No constant: each row carries the rate its host charged; unpinned runs are unpriced.


def label(r):
    return (f"{r['selector']}/{r['representation']}/{r.get('prompt', 'v1')}"
            f"{'/tl' if r.get('target_last') else ''}/{r.get('model', '?').split('/')[-1]}")


def load(directory):
    """{cell label: [(pair outcomes, rows), ...]} -- one entry per replicate."""
    cells = {}
    for path in sorted(Path(directory).glob("results_*.jsonl")):
        rows = [json.loads(line) for line in open(path)]
        if not rows:
            continue
        pairs = {}
        for r in rows:
            pairs.setdefault(r["pair_id"], {})[r["label"]] = r["prediction"]
        cells.setdefault(label(rows[0]), []).append((pairs, rows))
    return cells


def outcome(d):
    """P-C / P-V / P-B / P-R, or None when either half was unreadable."""
    v, b = d.get(1), d.get(0)
    if v is None or b is None or -1 in (v, b):
        return None
    return {(1, 0): "P-C", (1, 1): "P-V", (0, 0): "P-B", (0, 1): "P-R"}[(v, b)]


def mcnemar(a, b):
    """Two-sided exact McNemar on {pair: was P-C} for two cells."""
    keys = set(a) & set(b)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def task1(cells):
    """Each function on its own, replicates pooled; the bacc spread runs across replicates."""
    print("\ntask 1: absolute detection, each function on its own "
          "(always-VULNERABLE: acc 0.500, bacc 0.500, MCC 0)")
    print(f"{'cell':44} {'n':>4} {'vuln':>4} {'safe':>4} {'acc':>6} {'bacc':>6} "
          f"{'bacc spread':>13} {'MCC':>6} {'prec':>6} {'recall':>6} {'F1':>6} {'triv':>6} "
          f"{'AUPRC':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    for name, reps in cells.items():
        per = [compute_metrics(pd.DataFrame(rows))["balanced_accuracy"] for _, rows in reps]
        m = compute_metrics(pd.DataFrame([r for _, rows in reps for r in rows]))
        spread = f"[{min(per):.3f},{max(per):.3f}]" if len(per) > 1 else ""
        print(f"{name:44} {m['n_eval']:4d} {m['n_eval_vuln']:4d} {m['n_eval_safe']:4d} "
              f"{m['accuracy']:6.3f} {m['balanced_accuracy']:6.3f} {spread:>13} "
              f"{m['MCC']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['F1']:6.3f} "
              f"{m['F1_trivial']:6.3f} {m['AUPRC']:6.3f} {m['TP']:4d} {m['FP']:4d} {m['TN']:4d} "
              f"{m['FN']:4d}")


def operational(cells):
    """Failure-aware: the frozen set is the denominator, a failure counts as a wrong answer."""
    print("\noperational (failure-aware): denominator = the whole frozen set, "
          "every failure counts as a wrong answer")
    print(f"{'cell':44} {'n_set':>5} {'fail':>6} {'acc_op':>6} {'bacc_op':>7} {'MCC_op':>6} "
          f"{'F1_op':>6} {'triv':>6} {'pairs':>5} {'pAcc_op':>7} {'pAcc_op spread':>15}")
    for name, reps in cells.items():
        per = [compute_metrics(pd.DataFrame(rows))["pAcc_op"] for _, rows in reps]
        m = compute_metrics(pd.DataFrame([r for _, rows in reps for r in rows]))
        spread = f"[{min(per):.3f},{max(per):.3f}]" if len(per) > 1 else ""
        print(f"{name:44} {m['n_set']:5d} {m['failure_rate']:6.3f} {m['accuracy_op']:6.3f} "
              f"{m['balanced_accuracy_op']:7.3f} {m['MCC_op']:6.3f} {m['F1_op']:6.3f} "
              f"{m['F1_trivial_op']:6.3f} {m['n_pairs_set']:5d} {m['pAcc_op']:7.3f} {spread:>15}")


def main(directory, ref=None):
    cells = load(directory)
    if not cells:
        raise SystemExit(f"no results_*.jsonl under {directory}")
    correct = {}
    for name, reps in cells.items():
        correct[name] = {(i, p): outcome(d) == "P-C"
                         for i, (pairs, _) in enumerate(reps)
                         for p, d in pairs.items() if outcome(d) is not None}
    task1(cells)
    operational(cells)
    if ref:
        print(f"\ntask 2: paired discrimination   reference McNemar : {ref}"
              f"      (hasard apparie = 0.250)")
    else:
        print("\ntask 2: paired discrimination   (hasard apparie = 0.250)")
    print(f"{'cellule':44} {'rep':>3} {'n':>4} {'pAcc':>6} {'etendue':>13} {'P-C':>4} {'P-V':>4} "
          f"{'P-B':>4} {'P-R':>4} {'rank':>6} {'rank spread':>15} {'illis':>6} {'usd':>7}"
          + (f" {'b':>3} {'c':>3} {'McNemar':>8}" if ref else ""))
    total = 0.0
    for name, reps in cells.items():
        counts, bad, tin, tout, per_rep = {}, 0, 0, 0, []
        for pairs, rows in reps:
            hits = 0
            for d in pairs.values():
                o = outcome(d)
                counts[o] = counts.get(o, 0) + 1
                hits += o == "P-C"
            seen = sum(1 for d in pairs.values() if outcome(d) is not None)
            per_rep.append(hits / seen if seen else float("nan"))
            bad += sum(1 for r in rows if r["prediction"] == -1)
            tin += sum(r.get("billed_prompt_tokens") or 0 for r in rows)
            tout += sum(r.get("billed_completion_tokens") or 0 for r in rows)
        n = sum(v for k, v in counts.items() if k)
        pacc = counts.get("P-C", 0) / n if n else float("nan")
        spread = (f"[{min(per_rep):.3f},{max(per_rep):.3f}]" if len(per_rep) > 1 else "")
        rate_in = next((r.get("price_in") for _, rows in reps for r in rows
                        if r.get("price_in")), None)
        rate_out = next((r.get("price_out") for _, rows in reps for r in rows
                         if r.get("price_out")), None)
        usd = (tin * rate_in + tout * rate_out) if rate_in else float("nan")
        total += usd if usd == usd else 0.0
        ranks = [compute_metrics(pd.DataFrame(rows))["pair_rank_acc"] for _, rows in reps]
        rank = compute_metrics(pd.DataFrame([r for _, rows in reps for r in rows]))["pair_rank_acc"]
        rspread = f"[{min(ranks):.3f},{max(ranks):.3f}]" if len(ranks) > 1 else ""
        line = (f"{name:44} {len(reps):3d} {n:4d} {pacc:6.3f} {spread:>13} "
                f"{counts.get('P-C', 0):4d} {counts.get('P-V', 0):4d} "
                f"{counts.get('P-B', 0):4d} {counts.get('P-R', 0):4d} {rank:6.3f} {rspread:>15} "
                f"{bad:6d} {usd:7.4f}")
        if ref:
            n01, n10, p = mcnemar(correct[ref], correct[name])
            line += f" {n01:3d} {n10:3d} {p:8.3f}"
        print(line)
    print(f"\ntotal billed for this directory: {total:.4f} usd")
    print("spread = min and max pAcc across replicates of the SAME setting: any gap "
          "between cells smaller than that width is not a result.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
