"""The three phases end to end, unattended: python scripts/run_all.py --no-llm --subsample 2"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable, "-u"]     # unbuffered, so a container streams its log

# Used only when a phase promotes nothing, which is what a --no-llm rehearsal does.
FALLBACK_SELECTORS = ["S5", "S4", "S1"]
FALLBACK_PAIRS = [("S5", "R0"), ("S4", "R1"), ("S1", "R0"), ("S4", "R2")]

_log_path = None


def log(line=""):
    print(line, flush=True)
    if _log_path is not None:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sh(cmd):
    """Run a step, streaming its output to the console and the log; returns its exit code."""
    log(f"\n{'=' * 78}\n$ {' '.join(str(c) for c in cmd)}\n{'=' * 78}")
    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        log(line.rstrip("\n"))
    return proc.wait()


def promoted_from(path, fallback, phase):
    """The pairs a phase promoted, or a stated fallback, announced loudly."""
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8")).get("promoted", [])
        if rows:
            return [(r["selector"], r["representation"]) for r in rows]
    log(f"\n[run_all] WARNING: phase {phase} promoted nothing, falling back to {fallback}.")
    log("[run_all] Expected with --no-llm. In a real run it means no cell beat the C2 "
        "floor, and the next phase is NOT a valid experimental result.")
    return list(fallback)


def n_cells(out_dir):
    return len(list(Path(out_dir).glob("results_*.jsonl")))


def main():
    global _log_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "full_run"))
    ap.add_argument("--subsample", type=int, default=60)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--backend", choices=["ts", "joern"], default="ts")
    ap.add_argument("--model", default="google/gemma-3n-e4b-it")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default=os.environ.get("OCKHAM_API_KEY"))
    ap.add_argument("--provider", default=None)
    ap.add_argument("--prompt", default="v5")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top1", type=int, default=3, help="selectors promoted out of phase 1")
    ap.add_argument("--top2", type=int, default=4, help="pairs promoted out of phase 2")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _log_path = out / f"run_all_{time.strftime('%Y%m%d_%H%M%S')}.log"
    started = time.time()

    if not args.no_llm and not args.api_key:
        # Fail here, not in six hours: without a key every cell would be logged unavailable.
        sys.exit("no API key: pass --api-key or set OCKHAM_API_KEY, or run with --no-llm")

    log(f"[run_all] output {out}, budget {args.budget}, seed {args.seed}, "
        f"backend {args.backend}, {'no LLM' if args.no_llm else args.model}")

    common = ["--budget", args.budget, "--seed", args.seed, "--backend", args.backend,
              "--model", args.model, "--base-url", args.base_url, "--prompt", args.prompt,
              "--repeat", args.repeat, "--subsample", args.subsample]
    if args.api_key:
        common += ["--api-key", args.api_key]
    if args.provider:
        common += ["--provider", args.provider]
    if args.no_llm:
        common += ["--no-llm"]

    # One sample set for the three phases; run_experiment freezes it on the first phase.
    sample_set = out / "sample_set.json"
    gates = {}
    for phase, top in ((1, args.top1), (2, args.top2), (3, args.top2)):
        extra = []
        if phase == 2:
            pairs = promoted_from(out / "exp1" / "promoted.json",
                                  [(s, "R0") for s in FALLBACK_SELECTORS], 1)
            extra = ["--selectors"] + list(dict.fromkeys(s for s, _ in pairs))
        elif phase == 3:
            pairs = promoted_from(out / "exp2" / "promoted.json", FALLBACK_PAIRS, 2)
            extra = (["--selectors"] + [s for s, _ in pairs]
                     + ["--representations"] + [r for _, r in pairs])

        phase_dir = out / f"exp{phase}"
        sh(PY + ["scripts/run_experiment.py", "--phase", phase, "--sample-set", sample_set,
                 "--out-dir", phase_dir] + common + extra)
        log(f"\n[run_all] phase {phase}: {n_cells(phase_dir)} cell(s) with results")

        gates[phase] = sh(PY + ["scripts/validity_checks.py", phase_dir]) == 0
        sh(PY + ["scripts/promote.py", phase_dir, "--top", top,
                 "--json", phase_dir / "promoted.json"])

    log(f"\n{'=' * 78}\n[run_all] summary after {(time.time() - started) / 3600:.1f} h")
    for phase in (1, 2, 3):
        d = out / f"exp{phase}"
        state = "not started" if not d.exists() else f"{n_cells(d)} cell(s)"
        gate = "" if phase not in gates else ("  gates PASS" if gates[phase] else "  gates FAIL")
        log(f"  phase {phase}: {state}{gate}")
    log(f"  log: {_log_path}")
    log("  Rerun to retry anything missing: completed cells are skipped.")


if __name__ == "__main__":
    main()
