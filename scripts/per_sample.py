"""Per-sample outcome across runs: the functions the models always get right, or always miss."""

import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["project", "cwe", "runs", "acc", "unreadable", "p_vuln"]


def load(dirs):
    """Every results row below the dirs, once per (run_id, sample_id) so links never double count."""
    rows = [json.loads(line) for d in dirs for f in sorted(Path(d).rglob("results_*.jsonl"))
            for line in open(f, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(["run_id", "sample_id"])
    df["group"] = df.model.str.split("/").str[-1] + "/" + df.prompt
    return df


def per_sample(df):
    """One row per sample: share right among readable replies, overall and per model/prompt."""
    readable = df[df.prediction != -1]
    right = readable.prediction == readable.label
    out = df.groupby("sample_id").agg(
        pair_id=("pair_id", "first"), label=("label", "first"), project=("project", "first"),
        cwe=("cwe", lambda c: ",".join(c.iloc[0])), runs=("run_id", "size"),
        unreadable=("prediction", lambda p: float((p == -1).mean())),
        p_vuln=("p_vulnerable", "mean"))
    out["acc"] = right.groupby(readable.sample_id).mean()
    return out.join(right.groupby([readable.sample_id, readable.group]).mean().unstack())


def per_pair(df):
    """Share of runs in which both halves of a pair were right; an unreadable half counts wrong."""
    right = df.assign(right=df.prediction == df.label)
    return right.groupby(["run_id", "pair_id"]).right.all().groupby("pair_id").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="results directories, searched recursively")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", default=None, help="write the full per-sample table here")
    args = ap.parse_args()

    df = load(args.dirs)
    table = per_sample(df)
    print(f"{len(df)} rows, {df.run_id.nunique()} runs, {len(table)} samples; "
          f"groups: {', '.join(sorted(df.group.unique()))}")
    for label, name in ((1, "vulnerable"), (0, "benign")):
        # ties broken by p_vuln: a confident right answer ranks above a lucky one
        t = table[table.label == label].sort_values(["acc", "p_vuln"], ascending=[False, label == 0])
        print(f"\n{name}: most often right\n{t[COLS].head(args.top).to_string(float_format='%.2f')}")
        print(f"\n{name}: most often wrong\n"
              f"{t[COLS].tail(args.top).iloc[::-1].to_string(float_format='%.2f')}")
    pairs = per_pair(df)
    print(f"\npairs right in every run: {int((pairs == 1).sum())} of {len(pairs)}, "
          f"never right: {int((pairs == 0).sum())}")
    if args.csv:
        table.sort_values(["label", "acc"]).to_csv(args.csv, float_format="%.3f")
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
