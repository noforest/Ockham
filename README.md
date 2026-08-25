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

