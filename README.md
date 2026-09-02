# Ockham

> *Pluralitas non est ponenda sine necessitate.*
>
> William of Ockham

Building the smallest repository context an LLM needs to decide whether a C/C++
function is vulnerable.

## Dataset

```bash
mkdir data
cd data
curl -Lo pairs.jsonl https://raw.githubusercontent.com/alperen21/JitVul/main/data/final_benchmark.jsonl
```

## Usage

```python
# draw a sample set
python -m ockham.run --subsample 60 --seed 0 \
    --freeze-samples results/exp1/sample_set_60_pairs.json --freeze-only

# one cell without the model call: pack size and build time only
python -m ockham.run --selector S4 --representation R1 --limit 6 --no-llm

# a full phase: every selector on that frozen set, written to one directory
python scripts/run_experiment.py --phase 1 --repeat 3 --budget 2000 --seed 0 \
    --sample-set results/exp1/sample_set_60_pairs.json \
    --model <model> \
    --base-url <url> \
    --api-key "$API_KEY" \
    --out-dir results/exp1

# read the results data into nice tables
python scripts/show_metrics.py results/exp1
```

## Docker

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' $(id -u) $(id -g) > .env
docker compose build
```

Every command above runs unchanged inside the container:

```bash
docker compose run --rm ockham python -m ockham.run --selector S5 --representation R0 --limit 6 --no-llm
docker compose run --rm ockham python scripts/run_experiment.py --phase 1 --out-dir results/exp1 --no-llm
docker compose run -d --rm ockham python scripts/run_experiment.py --phase 1 --out-dir results/exp1
```


