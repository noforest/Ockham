"""Rank a finished phase and promote its best cells: python scripts/promote.py results/exp1 --top 3"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ockham.metrics import load_cells, spread

CONTROLS = {"C0", "C1", "C2"}
DEFAULT_TIE = 0.02


def _fmt(v, nd=3):
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def _rows(cells):
    rows, spreads = [], []
    for cell_id, (m, df, _path) in cells.items():
        _mean, std, n_rep = spread(df, "pAcc")
        if std == std and n_rep > 1:
            spreads.append(std)
        rows.append({"cell": cell_id, "selector": cell_id.split("_")[0],
                     "representation": cell_id.split("_")[1],
                     "pAcc": m.get("pAcc"), "MCC": m.get("MCC"),
                     "tokens": m.get("mean_evidence_tokens"),
                     "warm_ms": m.get("median_build_warm_ms"),
                     "unparsable": m.get("unparsable_rate"), "n_rep": n_rep})
    return rows, spreads


def tie_threshold(given, spreads):
    """The pAcc difference counted as a tie: measured from the replicates when there are any."""
    if given is not None:
        return given, "given"
    if spreads:
        return max(spreads), f"measured over {len(spreads)} repeated cell(s)"
    return DEFAULT_TIE, "default, no replicates found"


def promote(rows, floor, tie, top):
    """Highest pAcc first, cells within `tie` of a leader ordered by cost, floor applied."""
    ranked = sorted(rows, key=lambda r: -(r["pAcc"] if r["pAcc"] == r["pAcc"] else -1))
    pool = [r for r in ranked if r["selector"] not in CONTROLS and r["pAcc"] == r["pAcc"]
            and (floor is None or r["pAcc"] > floor)]
    promoted = []
    while pool and len(promoted) < top:
        leader = pool[0]
        group = [r for r in pool if leader["pAcc"] - r["pAcc"] <= tie]
        group.sort(key=lambda r: (_cost(r["tokens"]), _cost(r["warm_ms"])))
        for rank, r in enumerate(group[:top - len(promoted)]):
            r["why"] = ("highest pAcc" if len(group) == 1 else
                        f"tied with {leader['cell']} within {tie:.3f} pAcc, cheapest first"
                        if rank == 0 else f"tied with {leader['cell']}")
            promoted.append(r)
        pool = [r for r in pool if r not in group]
    return ranked, promoted


def _cost(v):
    return v if v == v else float("inf")


def _plain(v):
    """A number json can actually write: NaN is what a missing metric holds, and it is not JSON."""
    return None if v is None or v != v else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--tie-threshold", type=float, default=None)
    ap.add_argument("--json", default=None, help="write the promoted cells for the next phase")
    args = ap.parse_args()

    cells = load_cells(args.results_dir)
    if not cells:
        sys.exit(f"no results found in {args.results_dir}")

    rows, spreads = _rows(cells)
    tie, tie_src = tie_threshold(args.tie_threshold, spreads)
    floor = next((r["pAcc"] for r in rows if r["selector"] == "C2"), None)
    ranked, promoted = promote(rows, floor, tie, args.top)

    print(f"Promotion over {len(rows)} cells in {args.results_dir}")
    print(f"tie threshold = {tie:.3f} ({tie_src}); floor = C2 pAcc {_fmt(floor)}\n")
    print(f"{'cell':<20}{'pAcc':>8}{'MCC':>8}{'tokens':>9}{'warm ms':>10}{'rep':>5}  note")
    for r in ranked:
        note = "control" if r["selector"] in CONTROLS else ""
        if not note and floor is not None and r["pAcc"] == r["pAcc"] and r["pAcc"] <= floor:
            note = "below the C2 floor"
        print(f"{r['cell']:<20}{_fmt(r['pAcc']):>8}{_fmt(r['MCC']):>8}"
              f"{_fmt(r['tokens'], 0):>9}{_fmt(r['warm_ms'], 0):>10}{r['n_rep']:>5}  {note}")

    print("\nPromoted:")
    if not promoted:
        print("  none. No cell cleared the C2 floor, so nothing goes to the next phase.")
    for r in promoted:
        print(f"  {r['cell']:<20} pAcc {_fmt(r['pAcc'])}  tokens {_fmt(r['tokens'], 0)}"
              f"  ({r['why']})")
    print("\nSelectors for the next phase: "
          + " ".join(dict.fromkeys(r["selector"] for r in promoted)))

    if args.json:
        payload = {"results_dir": args.results_dir, "tie_threshold": _plain(tie),
                   "tie_source": tie_src, "floor_c2": _plain(floor),
                   "promoted": [{"cell": r["cell"], "selector": r["selector"],
                                 "representation": r["representation"],
                                 "pAcc": _plain(r["pAcc"]),
                                 "evidence_tokens": _plain(r["tokens"]), "reason": r["why"]}
                                for r in promoted]}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[promote] -> {args.json}")


if __name__ == "__main__":
    main()
