"""One experimental cell: (selector, representation, budget, sample set) -> one JSONL file.

    python -m ockham.run --selector C1 --representation R0 --limit 6

A cell applies the same treatment to every sample and writes one row per sample. No
experiment needs new code, only a different combination of those four values.

Per sample: checkout (only if the selector needs a pool) -> index -> build the pool ->
select under budget -> represent -> append a row. A checkout or index failure degrades
the pack to target-only and is counted, never raised.

Every row also carries what has to stay constant across a phase, so constancy can be
checked afterwards instead of being assumed.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import candidates as C
from .data import checkout, load_pairs
from .representation import r0_raw
from .selection import c0_target_only, c1_same_file, c2_random

SELECTORS = {
    "C0": c0_target_only.select,
    "C1": c1_same_file.select,
    "C2": c2_random.select,
}
REPRESENTATIONS = {"R0": r0_raw.render}

# Selectors that draw from the repository pool, and so need a checkout and an index.
NEEDS_POOL = {"C1", "C2"}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "pairs.jsonl"
WORKSPACE = ROOT / "workspace"


def _select_within_budget(selector_fn, sample, pool, budget):
    """Run the selector, then hold it to the budget here.

    Calling enforce_budget is a convention selectors are asked to follow; nothing stopped
    one from overshooting, and "the same budget for every cell" is what the comparison
    rests on. Checking at this level makes it true for any selector added later.
    """
    selected = selector_fn(sample, pool, budget)
    used = sum(c.tokens for c in selected)
    if used > budget:
        selected = C.enforce_budget(selected, budget)
        print(f"[warn] selector overshot budget ({used} > {budget}), clamped to "
              f"{sum(c.tokens for c in selected)}", flush=True)
    return selected


def build_pack(sample, selector_fn, represent_fn, needs_pool, budget):
    """Pack text plus the ledger fields for one sample."""
    target = sample.func_body.strip()
    t_cold = time.time()

    if not needs_pool:
        pack_text = represent_fn(target, [])
        warm_ms = (time.time() - t_cold) * 1000
        return _ledger(pack_text, [], 0, 0.0, warm_ms, warm_ms, 0)

    wt = checkout(sample, WORKSPACE)
    if wt is None:
        pack_text = represent_fn(target, [])       # degrade to target-only
        return _ledger(pack_text, [], 0, 0.0, (time.time() - t_cold) * 1000, 0.0, 1)

    index_s, n_symbols = C.ensure_indexed(wt)
    if n_symbols == 0:
        # The repository checked out but the backend found nothing in it. Counted as a
        # backend failure: passed off as an empty pool it would read exactly like "the
        # selector found nothing relevant".
        pack_text = represent_fn(target, [])
        return _ledger(pack_text, [], 0, index_s, (time.time() - t_cold) * 1000, 0.0, 1)

    pool = C.build_candidate_pool(sample, wt)
    t_warm = time.time()
    selected = _select_within_budget(selector_fn, sample, pool, budget)
    pack_text = represent_fn(target, selected)
    warm_ms = (time.time() - t_warm) * 1000
    return _ledger(pack_text, selected, len(pool), index_s,
                   (time.time() - t_cold) * 1000, warm_ms, 0)


def _ledger(pack_text, selected, pool_n, index_s, cold_ms, warm_ms, backend_fail):
    return {
        "pack_text": pack_text,
        "pack_tokens": C.count_tokens(pack_text),
        "n_evidence_tokens": sum(c.tokens for c in selected),
        "n_candidates_pool": pool_n,
        "n_candidates_selected": len(selected),
        "index_time_s": index_s,
        "build_time_cold_ms": cold_ms,
        "build_time_warm_ms": warm_ms,
        "n_backend_failures": backend_fail,
    }


@dataclass
class CellConfig:
    """Everything that identifies one cell. Drivers build this directly, no argparse."""
    selector: str = "C0"
    representation: str = "R0"
    budget: int = 2000
    backend: str = "ts"
    data: Path = DATA
    seed: int = 0
    limit: Optional[int] = None
    subsample: Optional[int] = None
    show_pack: bool = False
    out_dir: Optional[Path] = None

    def cell_id(self):
        # The backend belongs in the name: the two arms are distinct cells, and without it
        # they would overwrite each other's results file.
        return f"{self.selector}_{self.representation}_b{self.budget}_{self.backend}"


def resolve_samples(cfg):
    """The sample set for this cell."""
    path = Path(cfg.data)
    if not path.exists():
        raise SystemExit(f"no pair file at {path}. See the README for the expected format, "
                         f"or pass --data.")
    all_samples = load_pairs(path)
    if cfg.subsample:
        from .data import stratified_subsample
        return stratified_subsample(all_samples, n_pairs=cfg.subsample, seed=cfg.seed)
    if cfg.limit:
        return all_samples[: cfg.limit]
    return all_samples


def run_cell(cfg):
    """Execute one cell end to end. Returns the results path."""
    selector_fn = SELECTORS[cfg.selector]
    represent_fn = REPRESENTATIONS[cfg.representation]
    needs_pool = cfg.selector in NEEDS_POOL
    C.set_backend(cfg.backend)

    samples = resolve_samples(cfg)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{cfg.cell_id()}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_path = out_dir / f"results_{run_id}.jsonl"

    total = len(samples)
    with open(out_path, "w", encoding="utf-8") as out:
        for done, sample in enumerate(samples, 1):
            pack = build_pack(sample, selector_fn, represent_fn, needs_pool, cfg.budget)
            if cfg.show_pack:
                print(f"\n----- {sample.sample_id} -----\n{pack['pack_text']}")
            fail = " backend_failure" if pack["n_backend_failures"] else ""
            print(f"[{done}/{total}] {sample.sample_id} ({sample.project}) | {cfg.cell_id()} "
                  f"pool={pack['n_candidates_pool']} sel={pack['n_candidates_selected']} "
                  f"tokens={pack['pack_tokens']}{fail}", flush=True)

            record = {
                "sample_id": sample.sample_id, "pair_id": sample.pair_id, "cve": sample.cve,
                "cwe": sample.cwe, "project": sample.project, "commit": sample.commit,
                "label": sample.label,
                "run_id": run_id, "selector": cfg.selector,
                "representation": cfg.representation, "encoding": "text",
                "budget": cfg.budget, "backend": cfg.backend, "seed": cfg.seed,
                "target_tokens": C.count_tokens(sample.func_body),
                **{k: v for k, v in pack.items() if k != "pack_text"},
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    print(f"[run] {total} rows ({cfg.cell_id()}) -> {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", choices=list(SELECTORS), default="C0")
    ap.add_argument("--representation", choices=list(REPRESENTATIONS), default="R0")
    ap.add_argument("--backend", choices=sorted(C.BACKENDS), default="ts",
                    help="symbol backend, held constant within a phase")
    ap.add_argument("--budget", type=int, default=2000, help="token budget for the evidence")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="stratified subsample of N pairs by project and cwe")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--show-pack", action="store_true")
    args = ap.parse_args()

    run_cell(CellConfig(
        selector=args.selector, representation=args.representation, budget=args.budget,
        backend=args.backend, data=args.data, seed=args.seed, limit=args.limit,
        subsample=args.subsample, show_pack=args.show_pack, out_dir=args.out_dir,
    ))


if __name__ == "__main__":
    main()
