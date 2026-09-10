# Ockham

> *Pluralitas non est ponenda sine necessitate.* — William of Ockham

The smallest repository context an LLM needs to judge a C/C++ function.

## Install

```bash
pip install -r requirements.txt     # needs ctags and git on the PATH
```

## Data

`data/pairs.jsonl`, one JSON object per line. A line is a pair: the same function
before and after its fix.

| field | |
|---|---|
| `idx` | pair id |
| `cve`, `cwe`, `project`, `project_url`, `file_name` | provenance |
| `vulnerable_function_body`, `non_vulnerable_function_body` | the two halves |
| `vulnerable_commit_id`, `vulnerability_fixing_commit_id` | commits to check out |

```bash
python scripts/validate_dataset.py data/pairs.jsonl
```

## Run

```bash
# one cell, no model call
python -m ockham.run --selector S4 --representation R1 --limit 6 --no-llm

# one phase
export OCKHAM_API_KEY=<key>
python scripts/run_experiment.py --phase 1 --subsample 60 --out-dir results/exp1 \
    --model <model> --base-url <url>

# the three phases, unattended
python scripts/run_all.py --subsample 60 --model <model> --base-url <url> \
    --out-dir results/full_run

# read the results. 
python scripts/show_metrics.py <out-dir>
python scripts/compare_cells.py <phase-dir>

# read one exchange in full: parameters, system prompt, pack sent, reply.
python scripts/show_log.py <phase-dir>/logs/log_S4_R0_b2000_ts_r0_*.jsonl 1442-vuln
```

`<phase-dir>` is what a phase wrote: `--out-dir` itself for `run_experiment.py`, and
`<out-dir>/exp1`, `exp2`, `exp3` for `run_all.py`.

Every phase writes `<phase-dir>/logs/log_<run_id>.jsonl`, one JSON object per model call:
system prompt, the pack sent, the full reply, `finish_reason` and every parameter.
`<out-dir>/console_output_*.log` is a different thing, the driver's console transcript.

### Prompts

The system prompt lives in `prompts.toml`: pick an entry with `--prompt <name>` (`direct`,
`rationale`, their `*_cwe` variants, or `v5`). Each row records its hash (`prompt_sha`).
The verdict is read strictly from the `VERDICT:` line (`VULNERABLE` or `SAFE`, anything else
is -1), and unparsable replies are copied to `<phase-dir>/logs/unparsable_<run_id>.jsonl`.
Results come as two tasks, each function alone then the pairs, on the readable replies,
plus an operational view (`*_op`) where every failure counts as a wrong answer.

### What `run_all.py` does

1. **Freeze:** Draws the sample set once into `<out-dir>/sample_set.json`; all three
   phases use those same functions.
2. **Phase 1:** every selector at `R0`, fixed budget, into `<out-dir>/exp1/`.
3. **Gates:** `validity_checks.py` on that directory. Only what makes the numbers
   unreadable can stop the run -- unreadable replies, a suspiciously strong `C0`. A
   score below the always-vulnerable baseline is a result, not a fault, so it warns.
   Nothing fails under 10 pairs, where a single reply would decide. `--ignore-gates`
   runs anyway (`--no-llm` implies it, a rehearsal predicts nothing).
4. **Promotion:** `promote.py` ranks the cells by pAcc. A cell that does not beat the
   random control `C2` is never promoted. Cells within the tie threshold of the leader
   are ordered by cost, cheapest first, so a much lighter cell is not dropped over a
   difference inside the noise. The top `--top1` selectors go to phase 2.
5. **Phase 2:** those selectors x every representation, into `exp2/`; gates and
   promotion again, keeping the top `--top2` (selector, representation) pairs.
6. **Phase 3:** those pairs x budgets 2000 / 4000 / 8000, into `exp3/`.
7. **Summary:** cells and gate state per phase; everything is also in
   `<out-dir>/console_output_*.log`.

Rerunning the same command resumes: finished cells are skipped.

Selectors: `C0` target only, `C1` same file, `C2` random, `S1` BM25, `S2` dense,
`S3` fused, `S4` call graph, `S5` dependence slice.
Representations: `R0` raw, `R1` kept lines, `R2` call sites, `R3` structured
facts, `R4` primitive-API abstraction.

## Docker

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' $(id -u) $(id -g) > .env
docker compose build
docker compose run --rm ockham python scripts/run_all.py --subsample 60 --no-llm
```
