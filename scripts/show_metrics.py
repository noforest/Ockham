"""Print the metrics of one cell, or of every cell in a directory.

The metrics files store only medians, so every duration here is summed from the sibling
results_*.jsonl instead; a cell whose rows are missing shows "-".
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:                                  # so a copy of this file runs outside the repo too
    from ockham import candidates as C
    from ockham import solver
    SYSTEM_TOKENS = C.count_tokens(solver.SYSTEM_PROMPT)
except ImportError:
    SYSTEM_TOKENS = 120               # the v1 system prompt, measured; a fallback estimate

DETECTION = [("pairs", "n_pairs"), ("drop", "n_pairs_dropped_unparsable"),
             ("bkpair", "n_pairs_dropped_backend_failure"),
             ("kept", "n_samples_excluding_backend_failures"),
             # ("unprs", "unparsable_rate"), ("bkfail", "n_backend_failures_rate"),
             ("pAcc", "pAcc"), ("P-C", "P-C"), ("P-V", "P-V"), ("P-B", "P-B"),
             ("P-R", "P-R"), ("MCC", "MCC"), ("F1", "F1"), ("triv", "F1_trivial"),
             ("recall", "recall"), ("balanced_acc", "balanced_accuracy"),]
             # ("AUPRC", "AUPRC"), ("AUROC", "AUROC"), ("Brier", "Brier")]

# Cheapest of nine hosts, so a floor rather than the bill; the real rate is on each row.
MODEL, PRICE_IN, PRICE_OUT, MAX_TOKENS = "gemma-4-26b-a4b, cheapest host", 0.042, 0.220, 8

USD = [("in_tok", "usd_in_tokens"), ("out_tok", "usd_out_tokens"), ("usd", "usd_total")]

COST = [("pack", "mean_pack_tokens"), ("evid", "mean_evidence_tokens"),
        ("pool", "mean_candidates_pool"), ("sel", "mean_candidates_selected"),
        ("index_s", "median_index_time_s"), ("cold_ms", "median_build_cold_ms"),
        ("warm_ms", "median_build_warm_ms")]

TIME = [("started", "run_started"), ("build_s", "build_s"), ("of it index", "index_s"),
        ("llm_s", "llm_s"), ("other_s", "other_s"), ("elapsed_s", "wall_s")]


def _stamp(path):
    """The run_id timestamp the file name ends on, which is when the run started."""
    return datetime.strptime("_".join(path.stem.rsplit("_", 2)[-2:]), "%Y%m%d_%H%M%S")


def _hms(seconds):
    return f"{int(seconds) // 3600}h{int(seconds) % 3600 // 60:02d}m{int(seconds) % 60:02d}s"


def index_seconds(rows):
    """Indexing paid in this run: ensure_indexed replays its first timing on every later row."""
    seen, total = set(), 0.0
    for r in rows:
        key = (r.get("project"), r.get("commit"))
        if key not in seen:
            seen.add(key)
            total += r.get("index_time_s") or 0
    return total


def timings(metrics_path):
    """Durations for one run, summed from its rows: build_cold already contains index."""
    started = _stamp(metrics_path)
    out = {"run_started": started.strftime("%H:%M:%S"), "_started": started}
    rows_path = metrics_path.with_name(
        metrics_path.name.replace("metrics_", "results_", 1) + "l")
    if not rows_path.exists():
        return out
    rows = [json.loads(line) for line in open(rows_path)]
    if not rows:
        return out
    build = sum(r.get("build_time_cold_ms") or 0 for r in rows) / 1000
    llm = float(sum(r.get("llm_time_s") or 0 for r in rows))
    out.update(build_s=build, llm_s=llm, index_s=index_seconds(rows))
    return out


def wall_clock(cells):
    """Each run's wall clock = the gap to the next run's start. Sequential runs only."""
    order = sorted(cells, key=lambda c: c[1]["_started"])
    for (_, m), (_, nxt) in zip(order, order[1:]):
        m["wall_s"] = (nxt["_started"] - m["_started"]).total_seconds()
        m["other_s"] = m["wall_s"] - (m.get("build_s") or 0) - (m.get("llm_s") or 0)
    return cells


def load_cells(target):
    path = Path(target)
    files = sorted(path.rglob("metrics_*.json")) if path.is_dir() else [path]
    cells = []
    for f in files:
        try:                              # a killed run leaves a zero-byte metrics file
            metrics = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {f.name}: {e}")
            continue
        metrics.update(timings(f))
        cells.append((f.name.split("metrics_")[1].rsplit("_", 2)[0], metrics))
    if not cells:
        raise SystemExit(f"no readable metrics_*.json under {path}")
    return wall_clock(cells)


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
        sent = m.get("mean_prompt_tokens_total")
        if not (sent and sent == sent):        # a run from before the field existed
            sent = m["mean_pack_tokens"] + SYSTEM_TOKENS
        m["usd_in_tokens"] = int(sent * n)
        # the real reply length when the run recorded it; the v1 cap only as a fallback
        billed = m.get("mean_billed_completion_tokens")
        m["usd_out_tokens"] = int((billed if billed and billed == billed else MAX_TOKENS) * n)
        m["usd_total"] = f"{usd(m):.4f}"
    return cells


def _time_total(cells):
    """The one line to read: the three columns summed, and what they add up to."""
    timed = [m for _, m in cells if "wall_s" in m]
    if not timed:
        return
    build, llm, other, wall = (sum(m.get(k) or 0 for m in timed)
                               for k in ("build_s", "llm_s", "other_s", "wall_s"))
    print(f"\ntotal {_hms(build)} build + {_hms(llm)} llm + {_hms(other)} other "
          f"= {_hms(wall)} elapsed, over {len(timed)} of {len(cells)} runs")
    overlapped = [m for m in timed if (m.get("other_s") or 0) < 0]
    if overlapped:
        print(f"      {len(overlapped)} run(s) show a negative other_s: they overlapped, so "
              f"they were run in parallel and their elapsed_s is not a duration. The total "
              f"still holds -- the gaps tile the session either way.")
    starts = sorted(m["_started"] for _, m in cells)
    print(f"      {starts[0]:%Y-%m-%d %H:%M:%S} -> {starts[-1]:%Y-%m-%d %H:%M:%S} is the "
          f"first start to the LAST START; that last run's own duration is not recorded, "
          f"so the session ran a few minutes longer than the total above")


def main():
    cells = load_cells(sys.argv[1] if len(sys.argv) > 1 else "results")
    show(cells, DETECTION, "detection (s1)")
    show(cells, COST,
         "cost (s2 lightness, s3 scalability)   means over the samples of the run\n"
         "  pack     tokens actually sent: target function + context section\n"
         "  evid     of those, the selected context alone, which the budget caps\n"
         "  pool     tokens of every candidate the backend found in the checkout, before\n"
         "           selection. pool barely above the budget means the selector had little\n"
         "           left to choose: it keeps most of everything whatever its criterion\n"
         "  sel      how many functions were kept\n"
         "  index_s  MEDIAN time to index one checkout with tree-sitter + ctags\n"
         "  cold_ms  MEDIAN full build of one pack: checkout, index, pool, select, render.\n"
         "           What a fresh function costs, so this is the s3 headline\n"
         "  warm_ms  MEDIAN of select + render alone, on a checkout already indexed and\n"
         "           pooled. What a second function in the same repository costs. The gap\n"
         "           cold - warm is what the cache buys",
         column_width=10)
    show(cells, TIME,
         "time (s3)   build_s + llm_s + other_s = elapsed_s, exactly\n"
         "  started     when the run started (the run_id timestamp its file name ends on)\n"
         "  elapsed_s   gap to the next run's start AMONG THE RUNS LISTED HERE, so give the\n"
         "              script the whole session or the runs that ran in between land in it;\n"
         "              meaningless if you ran cells in parallel. The last run has no\n"
         "              successor, so it shows \"-\" and is left out of the total\n"
         "  build_s     sum of build_time_cold_ms: checkout, index, candidate pool,\n"
         "              selection, rendering -- everything before the API call\n"
         "  of it index the indexing inside build_s, counted once per distinct checkout\n"
         "              (each row replays the first timing, so summing the column triples it)\n"
         "  llm_s       sum of llm_time_s: time inside the API calls\n"
         "  other_s     elapsed_s - build_s - llm_s: loading the dataset, writing the\n"
         "              results, and any time the machine sat idle before the next run.\n"
         "              NEGATIVE means two runs overlapped in time, so they were not\n"
         "              sequential and elapsed_s is not that run's duration",
         column_width=11)
    _time_total(cells)
    show(with_cost(cells),
         USD,
         f"price (based on {MODEL}: ${PRICE_IN}/M input, ${PRICE_OUT}/M output)\n"
         f"  in_tok = mean sent prompt (else pack + {SYSTEM_TOKENS} system) * n_samples\n"
         f"  out_tok = mean billed reply (else {MAX_TOKENS}) * n_samples\n"
         f"  usd = in_tok / 1e6 * {PRICE_IN} + out_tok / 1e6 * {PRICE_OUT}",
         column_width=10)
    print(f"\ntotal {sum(usd(m) for _, m in cells):.4f} usd")


if __name__ == "__main__":
    main()
