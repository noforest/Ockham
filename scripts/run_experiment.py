"""Run one phase: the same sample set, budget and backend across a list of cells."""

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validity_checks import GATES, by_selector

from ockham import samples as S
from ockham.data import load_pairs
from ockham.metrics import load_cells
from ockham.run import DATA, REPRESENTATIONS, SELECTORS, CellConfig, run_cell

EXP1_SELECTORS = ["C0", "C1", "C2", "S1", "S2", "S3", "S4", "S5"]
EXP2_REPRESENTATIONS = ["R0", "R1", "R2", "R3", "R4"]
EXP3_BUDGETS = [2000, 4000, 8000]


def phase_cells(phase, selectors, representations, budget):
    """The (selector, representation, budget) triples one phase runs."""
    if phase == 1:
        selectors = selectors or EXP1_SELECTORS
        validate(selectors, ["R0"])
        return [(s, "R0", budget) for s in selectors]
    if phase == 2:
        if not selectors:
            sys.exit("--selectors is required for phase 2: the ones phase 1 promoted")
        representations = representations or EXP2_REPRESENTATIONS
        validate(selectors, representations)
        return [(s, r, budget) for s in selectors for r in representations]
    if phase == 3:
        if not selectors or not representations:
            sys.exit("--selectors and --representations are required for phase 3: the "
                     "pairs phase 2 promoted, in order")
        if len(selectors) != len(representations):
            sys.exit(f"phase 3 pairs --selectors with --representations positionally, so "
                     f"the two lists must have the same length (got {len(selectors)} and "
                     f"{len(representations)})")
        validate(selectors, representations)
        return [(s, r, b) for s, r in zip(selectors, representations) for b in EXP3_BUDGETS]
    sys.exit(f"unknown phase {phase}")


def validate(selectors, representations):
    """Reject unknown ids before any cell runs."""
    for ids, known, what in ((selectors, SELECTORS, "selector"),
                             (representations, REPRESENTATIONS, "representation")):
        unknown = [i for i in ids if i not in known]
        if unknown:
            sys.exit(f"unknown {what}(s): {', '.join(unknown)}. "
                     f"Known: {', '.join(sorted(known))}")


def freeze_once(path, data, subsample, seed):
    """Draw the phase's sample set unless it is already on disk."""
    if Path(path).exists():
        return
    ids = [s.sample_id for s in S.draw(load_pairs(data), n_pairs=subsample, seed=seed)]
    S.save(path, ids, seed, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--selectors", nargs="*", default=None,
                    help="phase 2 and 3: the selectors the previous phase promoted")
    ap.add_argument("--representations", nargs="*", default=None,
                    help="phase 2: defaults to every representation; phase 3: one per selector")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--backend", default="ts", help="held constant across the phase")
    ap.add_argument("--budget", type=int, default=2000, help="phases 1 and 2; phase 3 sweeps its own ladder")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--subsample", type=int, default=20, help="pairs, when freezing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-3n-e4b-it")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default=os.environ.get("OCKHAM_API_KEY"))
    ap.add_argument("--prompt", default="v5", help="held constant across the phase, "
                    "like the backend: it is part of what a cell is compared under")
    ap.add_argument("--provider", default=None,
                    help="pin one host for the whole phase; unpinned routing alone flips verdicts")
    ap.add_argument("--reasoning", choices=["off", "minimal", "low", "medium", "high"],
                    default=None, help="off on hosts that think by default and would "
                                       "otherwise spend the reply budget before writing")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun cells whose results are already on disk")
    ap.add_argument("--sample-set", default=None,
                    help="reuse a frozen set; defaults to one inside --out-dir")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cells = phase_cells(args.phase, args.selectors, args.representations, args.budget)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"exp{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_set = Path(args.sample_set) if args.sample_set else out_dir / "sample_set.json"
    freeze_once(sample_set, args.data, args.subsample, args.seed)

    print(f"[phase] {len(cells)} cells x {args.repeat} -> {out_dir}", flush=True)
    summary, failed = [], []
    for replicate in range(args.repeat):
        for selector, representation, budget in cells:
            cfg = CellConfig(
                selector=selector, representation=representation, budget=budget,
                backend=args.backend, data=args.data, model=args.model,
                base_url=args.base_url, api_key=args.api_key, no_llm=args.no_llm,
                prompt=args.prompt, provider=args.provider, reasoning=args.reasoning,
                seed=args.seed, replicate=replicate,
                sample_set=str(sample_set),
                out_dir=out_dir,
            )
            tag = f"{cfg.cell_id()} r{replicate}"
            done = list(out_dir.glob(f"results_{cfg.cell_id()}_r{replicate}_*.jsonl"))
            if done and not args.force:
                print(f"[phase] skip {tag}, results already at {done[-1].name}")
                continue
            print(f"[phase] {tag}")
            try:
                _path, metrics = run_cell(cfg)
            except Exception as e:                                      # noqa: BLE001
                print(f"[phase] ERROR {tag}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                failed.append(f"{tag} ({type(e).__name__})")
                continue
            summary.append((tag, metrics))

    print(f"\n[phase] {len(summary)} run, {len(failed)} unavailable", flush=True)
    for f in failed:
        print(f"  unavailable: {f}", flush=True)
    for cell_id, m in summary:
        print(f"[phase] {cell_id}: pAcc={m['pAcc']} MCC={m['MCC']} "
              f"F1={m['F1']} (trivial {m['F1_trivial']}) "
              f"pack_tokens={m['mean_pack_tokens']:.0f}")

    run_gates(out_dir)


def run_gates(out_dir):
    """The validity gates over what the phase wrote, so a bad phase says so on its own."""
    cells = load_cells(out_dir)
    if not cells:
        return
    print(f"\n[gates] {len(cells)} cells in {out_dir}")
    pacc = by_selector(cells)
    for name, gate in GATES:
        status, detail = gate(cells, pacc)
        print(f"  [{status}] {name}: {detail}", flush=True)


if __name__ == "__main__":
    main()
