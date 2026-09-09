"""The gates a phase must pass before its numbers are read: python scripts/validity_checks.py results/exp1"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ockham.metrics import load_cells

MAX_C0_PACC = 0.85        # above this, C0 alone is recalling the fix, not analysing
MAX_UNPARSABLE = 0.05
MIN_PAIRS = 10            # below this a gate would be failing on noise
CONTROLS = {"C0", "C1", "C2"}


def _fmt(v, nd=3):
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def _val(v):
    """None for a missing or nan metric, so a gate says SKIP instead of passing silently."""
    return None if v is None or v != v else v


def by_selector(cells, metric="pAcc"):
    out = {}
    for cell_id, (metrics, _df, _p) in cells.items():
        out.setdefault(cell_id.split("_")[0], _val(metrics.get(metric)))
    return out


def contamination(cells, pacc):
    """C0 carries no context, so a high score there is memorisation rather than detection."""
    c0 = pacc.get("C0")
    if c0 is None:
        return "SKIP", "needs a C0 cell with a score"
    if c0 > MAX_C0_PACC:
        return "FAIL", f"C0 pAcc {_fmt(c0)} > {MAX_C0_PACC} with no context at all"
    return "PASS", f"C0 pAcc {_fmt(c0)} <= {MAX_C0_PACC}"


def random_control(cells, pacc):
    """A selector that does not beat C2 fills the budget rather than selecting."""
    c2 = pacc.get("C2")
    if c2 is None:
        return "SKIP", "needs a C2 cell with a score"
    below = sorted(s for s, v in pacc.items()
                   if s not in CONTROLS and v is not None and v <= c2)
    if below:
        return "WARN", f"C2 = {_fmt(c2)}, not beaten by {', '.join(below)}"
    return "PASS", f"C2 = {_fmt(c2)}, beaten by every selector"


def trivial_baseline(cells, _pacc):
    """F1 under the always-vulnerable classifier is worse than answering nothing."""
    worse = [f"{cid} (F1 {_fmt(m.get('F1'))} < {_fmt(m.get('F1_trivial'))})"
             for cid, (m, _df, _p) in sorted(cells.items())
             if _val(m.get("F1")) is not None and _val(m.get("F1_trivial")) is not None
             and m["F1"] < m["F1_trivial"]]
    # A result, not an invalidity: reported, never blocking.
    if worse:
        return "WARN", f"{len(worse)} cell(s) below trivial: {'; '.join(worse[:4])}"
    return "PASS", "every cell at or above the always-vulnerable F1"


def unreadable(cells, _pacc):
    """Above a few percent of unreadable replies the cell measures the endpoint, not the pack."""
    noisy = [f"{cid} ({_fmt(m.get('unparsable_rate'), 2)})"
             for cid, (m, _df, _p) in sorted(cells.items())
             if (m.get("unparsable_rate") or 0) > MAX_UNPARSABLE]
    if noisy:
        return "FAIL", f"{len(noisy)} cell(s) above {MAX_UNPARSABLE}: {'; '.join(noisy[:4])}"
    return "PASS", f"every cell at or below {MAX_UNPARSABLE}"


GATES = [("contamination probe", contamination), ("C2 random control", random_control),
         ("trivial baseline", trivial_baseline), ("unreadable answers", unreadable)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=str(ROOT / "results"))
    args = ap.parse_args()

    cells = load_cells(args.results_dir)
    if not cells:
        sys.exit(f"no results found in {args.results_dir}")

    n_pairs = max((m.get("n_pairs") or 0 for m, _df, _p in cells.values()), default=0)
    print(f"Validity checks over {len(cells)} cells in {args.results_dir}\n")
    pacc = by_selector(cells)
    failed = warned = 0
    for name, gate in GATES:
        status, detail = gate(cells, pacc)
        if status == "FAIL" and n_pairs < MIN_PAIRS:
            status, detail = "WARN", f"{detail} -- only {n_pairs} pair(s), too few to fail on"
        failed += status == "FAIL"
        warned += status == "WARN"
        print(f"  [{status}] {name}: {detail}", flush=True)
    print(f"\n{failed} gate(s) failed, {warned} warning(s)." if failed
          else f"\nNo blocking gate, {warned} warning(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
