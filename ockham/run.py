"""One cell -- selector, representation, budget, sample set -- to one JSONL file."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import candidates as C
from . import embeddings
from . import samples as S
from . import solver
from .data import checkout, load_pairs
from .metrics import compute_metrics, load_results
from .representation import r0_raw, r1_snippets
from .selection import (c0_target_only, c1_same_file, c2_random, s1_bm25, s2_dense,
                        s3_hybrid, s4_callgraph, s5_slice)

SELECTORS = {
    "C0": c0_target_only.select,
    "C1": c1_same_file.select,
    "C2": c2_random.select,
    "S1": s1_bm25.select,
    "S2": s2_dense.select,
    "S3": s3_hybrid.select,
    "S4": s4_callgraph.select,
    "S5": s5_slice.select,
}
REPRESENTATIONS = {"R0": r0_raw.render, "R1": r1_snippets.render}

NEEDS_POOL = {"C1", "C2", "S1", "S2", "S3", "S4", "S5"}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "pairs.jsonl"
WORKSPACE = ROOT / "workspace"


def _select_within_budget(selector_fn, sample, pool, budget):
    """Run the selector, then hold it to the budget here rather than trust it to."""
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
        # An empty pool would read exactly like a selector finding nothing relevant.
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
    model: str = "google/gemma-3n-e4b-it"
    base_url: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None
    no_llm: bool = False
    logprobs: bool = True
    max_tokens: int = solver.DEFAULT_MAX_TOKENS
    reasoning: Optional[str] = None
    seed: int = 0
    replicate: int = 0
    limit: Optional[int] = None
    subsample: Optional[int] = None
    sample_set: Optional[str] = None
    freeze_samples: Optional[str] = None
    freeze_only: bool = False
    show_pack: bool = False
    out_dir: Optional[Path] = None

    def cell_id(self):
        # The backend belongs in the name, or the two arms overwrite each other.
        return f"{self.selector}_{self.representation}_b{self.budget}_{self.backend}"


def resolve_samples(cfg):
    """The sample set for this cell, and its id."""
    path = Path(cfg.data)
    if not path.exists():
        raise SystemExit(f"no pair file at {path}. See the README for the expected format, "
                         f"or pass --data.")
    all_samples = load_pairs(path)
    if cfg.sample_set:
        payload = S.load(cfg.sample_set, path)
        return S.apply(all_samples, payload["sample_ids"]), payload["sample_set_id"]

    if cfg.subsample:
        chosen = S.draw(all_samples, n_pairs=cfg.subsample, seed=cfg.seed)
    elif cfg.limit:
        chosen = all_samples[: cfg.limit]
    else:
        chosen = all_samples
    ids = [s.sample_id for s in chosen]
    if cfg.freeze_samples:
        S.save(cfg.freeze_samples, ids, cfg.seed, path)
    return chosen, S.sample_set_id(ids)


def run_cell(cfg):
    """Execute one cell end to end. Returns the results path."""
    selector_fn = SELECTORS[cfg.selector]
    represent_fn = REPRESENTATIONS[cfg.representation]
    needs_pool = cfg.selector in NEEDS_POOL
    C.set_backend(cfg.backend)

    samples, set_id = resolve_samples(cfg)
    if cfg.freeze_only:
        return None
    out_dir = Path(cfg.out_dir) if cfg.out_dir else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{cfg.cell_id()}_r{cfg.replicate}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_path = out_dir / f"results_{run_id}.jsonl"

    total = len(samples)
    with open(out_path, "w", encoding="utf-8") as out:
        for done, sample in enumerate(samples, 1):
            pack = build_pack(sample, selector_fn, represent_fn, needs_pool, cfg.budget)
            if cfg.show_pack:
                print(f"\n----- {sample.sample_id} -----\n{pack['pack_text']}")
            if cfg.no_llm:
                prediction, p_vulnerable, raw, billed, llm_s = -1, None, "[no-llm]", None, 0.0
            else:
                t_llm = time.time()
                prediction, p_vulnerable, raw, billed = solver.predict(
                    pack["pack_text"], cfg.model, cfg.base_url, cfg.api_key, seed=cfg.seed,
                    logprobs=cfg.logprobs, max_tokens=cfg.max_tokens,
                    reasoning=cfg.reasoning)
                llm_s = time.time() - t_llm

            pred_str = {1: "VULN", 0: "SAFE", -1: "??"}[prediction]
            mark = "??" if prediction == -1 else ("OK" if prediction == sample.label else "FAIL")
            fail = " backend_failure" if pack["n_backend_failures"] else ""
            print(f"[{done}/{total}] {sample.sample_id} ({sample.project}) | {cfg.cell_id()} "
                  f"pool={pack['n_candidates_pool']} sel={pack['n_candidates_selected']} "
                  f"tokens={pack['pack_tokens']} | pred={pred_str} "
                  f"label={'VULN' if sample.label else 'SAFE'} [{mark}]{fail}", flush=True)

            record = {
                "sample_id": sample.sample_id, "pair_id": sample.pair_id, "cve": sample.cve,
                "cwe": sample.cwe, "project": sample.project, "commit": sample.commit,
                "label": sample.label,
                "run_id": run_id, "sample_set_id": set_id, "selector": cfg.selector,
                "representation": cfg.representation, "encoding": "text",
                "budget": cfg.budget, "backend": cfg.backend, "seed": cfg.seed,
                "replicate": cfg.replicate,
                "model": cfg.model, "base_url": cfg.base_url,
                "reasoning": cfg.reasoning, "max_tokens": cfg.max_tokens,
                "s2_model": embeddings.MODEL_NAME,
                "prediction": prediction, "p_vulnerable": p_vulnerable,
                "model_output_raw": raw, "llm_time_s": llm_s,
                "billed_prompt_tokens": (billed or {}).get("prompt"),
                "billed_completion_tokens": (billed or {}).get("completion"),
                "billed_reasoning_tokens": (billed or {}).get("reasoning"),
                "target_tokens": C.count_tokens(sample.func_body),
                "prompt_tokens_total": C.count_tokens(solver.SYSTEM_PROMPT) + pack["pack_tokens"],
                **{k: v for k, v in pack.items() if k != "pack_text"},
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    print(f"[run] {total} rows ({cfg.cell_id()}, set {set_id}) -> {out_path}", flush=True)

    metrics = compute_metrics(load_results(out_path))
    metrics_path = out_dir / f"metrics_{run_id}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    for k, v in metrics.items():
        print(f"[metrics] {k}: {v}", flush=True)
    print(f"[metrics] -> {metrics_path}", flush=True)
    return out_path, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", choices=list(SELECTORS), default="C0")
    ap.add_argument("--representation", choices=list(REPRESENTATIONS), default="R0")
    ap.add_argument("--backend", choices=sorted(C.BACKENDS), default="ts",
                    help="symbol backend, held constant within a phase")
    ap.add_argument("--budget", type=int, default=2000, help="token budget for the evidence")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--model", default="google/gemma-3n-e4b-it")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default=os.environ.get("OCKHAM_API_KEY"))
    ap.add_argument("--reasoning", choices=["off", "minimal", "low", "medium", "high"],
                    default=None,
                    help="thinking budget on routers that expose one; unset sends nothing")
    ap.add_argument("--max-tokens", type=int, default=solver.DEFAULT_MAX_TOKENS,
                    help="reply cap; only the first word is read, but a reasoning model "
                         "spends its whole budget before emitting a verdict")
    ap.add_argument("--no-logprobs", action="store_true",
                    help="hard verdict only: pAcc/MCC/F1 stay, rank/AUROC/Brier go away. "
                         "For endpoints with no logprob-capable provider.")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the detection call: prediction = -1 on every row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="stratified subsample of N pairs by project and cwe")
    ap.add_argument("--sample-set", default=None,
                    help="reuse a frozen sample set (overrides --limit/--subsample)")
    ap.add_argument("--freeze-samples", default=None,
                    help="write the drawn set to this path for later cells to reuse")
    ap.add_argument("--freeze-only", action="store_true",
                    help="with --freeze-samples: write the set and stop, running no cell")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--show-pack", action="store_true")
    args = ap.parse_args()
    if args.freeze_only and not args.freeze_samples:
        ap.error("--freeze-only needs --freeze-samples to say where to write the set")

    run_cell(CellConfig(
        selector=args.selector, representation=args.representation, budget=args.budget,
        backend=args.backend, data=args.data, model=args.model, base_url=args.base_url,
        logprobs=not args.no_logprobs, max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        api_key=args.api_key, no_llm=args.no_llm, seed=args.seed, limit=args.limit,
        subsample=args.subsample, sample_set=args.sample_set,
        freeze_samples=args.freeze_samples, freeze_only=args.freeze_only,
        show_pack=args.show_pack, out_dir=args.out_dir,
    ))


if __name__ == "__main__":
    main()
