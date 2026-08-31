# Ockham

> *Pluralitas non est ponenda sine necessitate.*
>
> William of Ockham

Building the smallest repository context an LLM needs to decide whether a C/C++
function is vulnerable.

## Usage

```python
# draw a sample set
python -m ockham.run --selector C0 --representation R0 --subsample 20 \
    --freeze-samples results/exp1/sample_set.json --freeze-only

# one cell without the model call: pack size and build time only
python -m ockham.run --selector S4 --representation R1 --limit 6 --no-llm

# a full phase: every selector on that frozen set, written to one directory
python scripts/run_experiment.py --phase 1 --model <model> --base-url <url> --out-dir results/exp1

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
```

The embedding model and the tokenizer are baked into the image, so a run needs the
network only to clone the repositories under test. `OCKHAM_API_KEY` reaches the model
endpoint through the environment or `.env`; `OCKHAM_ROOT` chooses which host directory
is mounted. Detach a long run so it outlives the ssh session:

```bash
docker compose run -d --rm ockham python scripts/run_experiment.py --phase 1 --out-dir results/exp1
```

