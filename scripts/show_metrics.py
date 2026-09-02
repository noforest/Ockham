"""Print the metrics of one cell, or of every cell in a directory."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ockham import candidates as C
from ockham import solver

DETECTION = [("pairs", "n_pairs"), ("drop", "n_pairs_dropped_unparsable"),
             ("bkpair", "n_pairs_dropped_backend_failure"),
             ("kept", "n_samples_excluding_backend_failures"),
             # ("unprs", "unparsable_rate"), ("bkfail", "n_backend_failures_rate"),
             ("pAcc", "pAcc"), ("P-C", "P-C"), ("P-V", "P-V"), ("P-B", "P-B"),
             ("P-R", "P-R"), ("MCC", "MCC"), ("F1", "F1"), ("triv", "F1_trivial"),
             ("recall", "recall"), ("balanced_acc", "balanced_accuracy")]

MODEL, PRICE_IN, PRICE_OUT, MAX_TOKENS = "gemma-4-26b-a4b", 0.042, 0.220, 8
SYS = C.count_tokens(solver.SYSTEM_PROMPT)

USD = [("in_tok", "usd_in_tokens"), ("out_tok", "usd_out_tokens"), ("usd", "usd_total")]

COST = [("pack", "mean_pack_tokens"), ("evid", "mean_evidence_tokens"),
        ("pool", "mean_candidates_pool"), ("sel", "mean_candidates_selected"),
        ("index_s", "median_index_time_s"), ("cold_ms", "median_build_cold_ms"),
        ("warm_ms", "median_build_warm_ms")]


def load_cells(target):
    path = Path(target)
    files = sorted(path.glob("metrics_*.json")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no metrics_*.json under {path}")
    return [(f.name.split("metrics_")[1].rsplit("_", 2)[0], json.loads(f.read_text()))
            for f in files]


def show(cells, columns, title, column_width=8):
    width = max(len(name) for name, _ in cells)
    print(f"\n{title}")
    print(f"{'cell':<{width}} " + " ".join(f"{label:>{column_width}}" for label, _ in columns))
    for name, metrics in cells:
        values = []
        for _, key in columns:
            value = metrics.get(key)
            if value is None:
                value = "-"
            values.append(f"{value:>{column_width}.3f}" if isinstance(value, float)
                          else f"{value:>{column_width}}")
        print(f"{name:<{width}} " + " ".join(values))


def usd(m):
    return (m["usd_in_tokens"] * PRICE_IN + m["usd_out_tokens"] * PRICE_OUT) / 1e6


def with_cost(cells):
    for _, m in cells:
        n = m["n_samples"]
        m["usd_in_tokens"] = int((m["mean_pack_tokens"] + SYS) * n)
        m["usd_out_tokens"] = int(MAX_TOKENS * n)
        m["usd_total"] = f"{usd(m):.4f}"
    return cells


def main():
    cells = load_cells(sys.argv[1] if len(sys.argv) > 1 else "results")
    show(cells, DETECTION, "detection (s1)")
    show(cells, COST, "cost (s2, s3)", column_width=10)
    show(with_cost(cells),
         USD,
         f"price (based on {MODEL}: ${PRICE_IN}/M input, ${PRICE_OUT}/M output)\n"
         f"  in_tok = (pack + {SYS} system) * n_samples\n"
         f"  out_tok = {MAX_TOKENS} * n_samples\n"
         f"  usd = in_tok / 1e6 * {PRICE_IN} + out_tok / 1e6 * {PRICE_OUT}",
         column_width=10)
    print(f"\ntotal {sum(usd(m) for _, m in cells):.4f} usd")


if __name__ == "__main__":
    main()
