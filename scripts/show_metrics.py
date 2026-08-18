"""Print the metrics of one cell, or of every cell in a directory."""

import json
import sys
from pathlib import Path

DETECTION = [("pairs", "n_pairs"), ("drop", "n_pairs_dropped_unparsable"),
             ("unprs", "unparsable_rate"), ("bkfail", "n_backend_failures_rate"),
             ("pAcc", "pAcc"), ("P-C", "P-C"), ("P-V", "P-V"), ("P-B", "P-B"),
             ("P-R", "P-R"), ("MCC", "MCC"), ("F1", "F1"), ("triv", "F1_trivial"),
             ("recall", "recall")]

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
            values.append(f"{value:>{column_width}.3f}" if isinstance(value, float)
                          else f"{value:>{column_width}}")
        print(f"{name:<{width}} " + " ".join(values))


def main():
    cells = load_cells(sys.argv[1] if len(sys.argv) > 1 else "results")
    show(cells, DETECTION, "detection (s1)")
    show(cells, COST, "cost (s2, s3)", column_width=10)


if __name__ == "__main__":
    main()
