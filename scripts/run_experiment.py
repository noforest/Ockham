"""Run one phase: the same sample set and budget across a list of cells.

    python scripts/run_experiment.py --phase 1 --selectors C0 C2 S1 --no-llm

Phase 1 varies the selector at a fixed representation, so the backend sits on the
phase and never on a single cell: a selector that also changed the backend would win
on two counts at once with no way to separate them.

A cell whose results file already exists is skipped, which is what makes the loop
resumable. Nothing else carries state.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ockham import samples as S                                   # noqa: E402
from ockham.data import load_pairs                                # noqa: E402
from ockham.run import DATA, CellConfig, run_cell                 # noqa: E402

PHASES = {1: {"representations": ["R0"]}}


def freeze_once(path, data, subsample, seed):
    """Draw the phase's sample set unless it is already on disk."""
    if Path(path).exists():
        return
    ids = [s.sample_id for s in S.draw(load_pairs(data), n_pairs=subsample, seed=seed)]
    S.save(path, ids, seed, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=sorted(PHASES), default=1)
    ap.add_argument("--selectors", nargs="+", default=["C0", "C2", "S1"])
    ap.add_argument("--backend", default="ts", help="held constant across the phase")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--subsample", type=int, default=20, help="pairs, when freezing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-3n-e4b-it")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"exp{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_set = out_dir / "sample_set.json"
    freeze_once(sample_set, args.data, args.subsample, args.seed)

    summary = []
    for selector in args.selectors:
        for representation in PHASES[args.phase]["representations"]:
            cfg = CellConfig(
                selector=selector, representation=representation, budget=args.budget,
                backend=args.backend, data=args.data, model=args.model,
                base_url=args.base_url, api_key=args.api_key, no_llm=args.no_llm,
                seed=args.seed, sample_set=str(sample_set), out_dir=out_dir,
            )
            done = list(out_dir.glob(f"results_{cfg.cell_id()}_*.jsonl"))
            if done:
                print(f"[phase] skip {cfg.cell_id()}, results already at {done[-1].name}")
                continue
            print(f"[phase] {cfg.cell_id()}")
            _path, metrics = run_cell(cfg)
            summary.append((cfg.cell_id(), metrics))

    for cell_id, m in summary:
        print(f"[phase] {cell_id}: pAcc={m['pAcc']} MCC={m['MCC']} "
              f"F1={m['F1']} (trivial {m['F1_trivial']}) "
              f"pack_tokens={m['mean_pack_tokens']:.0f}")


if __name__ == "__main__":
    main()
