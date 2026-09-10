"""Render a logs/log_*.jsonl as text, or rebuild the pack of an older results_*.jsonl."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ockham import candidates as C
from ockham import run as R
from ockham import prompts
from ockham.data import load_pairs

RULE = "=" * 78


def rebuild_pack(row):
    """The pack text of an old row, rebuilt from the fields the row already records."""
    C.set_backend(row["backend"])
    sample = next(s for s in load_pairs(R.DATA) if s.sample_id == row["sample_id"])
    represent = partial(R.REPRESENTATIONS[row["representation"]],
                        target_last=row.get("target_last", False))
    pack = R.build_pack(sample, R.SELECTORS[row["selector"]], represent,
                        row["selector"] in R.NEEDS_POOL, row["budget"])
    return pack["pack_text"], pack["evidence_names"]


def render(row, system_prompt=None):
    if "user_message" in row:                    # a log row carries the pack already
        pack_text, names = row["user_message"], row.get("evidence_names", [])
        reply = row["response"]
    else:
        pack_text, names = rebuild_pack(row)
        reply = row["model_output_raw"]
    params = {k: row.get(k) for k in (
        "run_id", "sample_id", "pair_id", "label", "cve", "cwe", "project", "commit",
        "model", "base_url", "provider", "temperature", "max_tokens", "seed", "reasoning",
        "prompt_id", "prompt", "prompt_sha", "selector", "representation", "budget", "backend",
        "replicate", "sample_set_id", "finish_reason", "prediction", "p_vulnerable", "cwe_pred", "llm_time_s",
        "billed_prompt_tokens", "billed_completion_tokens", "pack_tokens",
        "n_evidence_tokens", "n_candidates_pool", "n_candidates_selected")
        if row.get(k) is not None}
    params.setdefault("temperature", 0)
    out = [RULE, f"LOG  {row['sample_id']}  ({row.get('run_id', '?')})", RULE,
           "\n-- PARAMETERS " + "-" * 64,
           json.dumps(params, indent=2, default=str),
           f"\nevidence kept ({len(names)}): {', '.join(names) or '(none)'}",
           "\n-- SYSTEM PROMPT " + "-" * 61,
           system_prompt or row.get("system_prompt")
           or prompts.load(row.get("prompt_id") or row.get("prompt", "v5")).text,
           "\n-- USER MESSAGE (the pack) " + "-" * 51, pack_text,
           "\n-- MODEL REPLY " + "-" * 63, reply, ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="a logs/log_*.jsonl or a results_*.jsonl file")
    ap.add_argument("sample_ids", nargs="*", help="default: every row in the file")
    ap.add_argument("-o", "--out", default=None, help="write here instead of stdout")
    ap.add_argument("--system-prompt-file", default=None,
                    help="the prompt as it stood at run time, when prompts.toml has moved since")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    if args.sample_ids:
        rows = [r for r in rows if r["sample_id"] in args.sample_ids]
    sys_prompt = (Path(args.system_prompt_file).read_text(encoding="utf-8")
                  if args.system_prompt_file else None)
    text = "\n".join(render(r, sys_prompt) for r in rows)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[log] {len(rows)} entries -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
