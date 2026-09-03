"""Compare cells: pAcc, the pair breakdown, cost, McNemar. Replicates are pooled.

    python scripts/compare_cells.py results/grid [reference_cell]
"""

import json
import sys
from math import comb
from pathlib import Path

# No constant: each row carries the rate its host charged; unpinned runs are unpriced.


def label(r):
    return (f"{r['selector']}/{r['representation']}/{r.get('prompt', 'v1')}"
            f"{'/tl' if r.get('target_last') else ''}")


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


def main(directory, ref=None):
    cells = load(directory)
    if not cells:
        raise SystemExit(f"no results_*.jsonl under {directory}")
    correct = {}
    for name, reps in cells.items():
        correct[name] = {(i, p): outcome(d) == "P-C"
                         for i, (pairs, _) in enumerate(reps)
                         for p, d in pairs.items() if outcome(d) is not None}
    ref = ref or next(iter(cells))

    print(f"\nreference McNemar : {ref}      (hasard apparie = 0.250)")
    print(f"{'cellule':22} {'rep':>3} {'n':>4} {'pAcc':>6} {'etendue':>13} {'P-C':>4} {'P-V':>4} "
          f"{'P-B':>4} {'P-R':>4} {'illis':>6} {'usd':>7} {'b':>3} {'c':>3} {'McNemar':>8}")
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
        n01, n10, p = mcnemar(correct[ref], correct[name])
        print(f"{name:22} {len(reps):3d} {n:4d} {pacc:6.3f} {spread:>13} {counts.get('P-C',0):4d} "
              f"{counts.get('P-V',0):4d} {counts.get('P-B',0):4d} {counts.get('P-R',0):4d} "
              f"{bad:6d} {usd:7.4f} {n01:3d} {n10:3d} {p:8.3f}")
    print(f"\ntotal facture sur ce dossier : {total:.4f} usd")
    print("etendue = pAcc min et max entre replicats du MEME reglage : tout ecart entre "
          "cellules plus petit que cette largeur n'est pas un resultat.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
