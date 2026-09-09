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
python scripts/run_experiment.py --phase 1 --subsample 60 --out-dir results/exp1 \
    --model <model> --base-url <url> --api-key "$KEY"

# the three phases, unattended
python scripts/run_all.py --subsample 60 --model <model> --base-url <url>

# read the results
python scripts/show_metrics.py results/exp1
```

Selectors: `C0` target only, `C1` same file, `C2` random, `S1` BM25, `S2` dense,
`S3` fused, `S4` call graph, `S5` dependence slice.
Representations: `R0` raw, `R1` kept lines, `R2` call sites.

## Docker

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' $(id -u) $(id -g) > .env
docker compose build
docker compose run --rm ockham python scripts/run_all.py --subsample 60 --no-llm
```
