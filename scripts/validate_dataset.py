"""Check a pair file against the data contract: python scripts/validate_dataset.py data/pairs.jsonl"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ockham.data import load_pairs

REQUIRED = ["idx", "cve", "cwe", "project", "project_url", "file_name",
            "vulnerable_function_body", "non_vulnerable_function_body",
            "vulnerable_commit_id", "vulnerability_fixing_commit_id"]


def check_lines(path):
    """Problems that break the contract, and repeated idx, which the loader handles."""
    problems, repeats, seen, n = [], [], set(), 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {i}: not JSON ({e.msg})")
                continue
            missing = [k for k in REQUIRED if k not in row]
            if missing:
                problems.append(f"line {i}: missing {', '.join(missing)}")
                continue
            if not isinstance(row["cwe"], list):
                problems.append(f"line {i}: cwe is {type(row['cwe']).__name__}, expected a list")
            if row["idx"] in seen:
                repeats.append(row["idx"])
            seen.add(row["idx"])
    return n, len(seen), problems, repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(ROOT / "data" / "pairs.jsonl"))
    ap.add_argument("--show", type=int, default=10, help="how many problems to print")
    args = ap.parse_args()

    if not Path(args.path).exists():
        sys.exit(f"no file at {args.path}")

    n_lines, n_idx, problems, repeats = check_lines(args.path)
    print(f"{args.path}: {n_lines} lines, {n_idx} distinct idx")
    if repeats:
        print(f"  note: {len(repeats)} repeated idx, the loader keeps the first of each")
    for p in problems[:args.show]:
        print(f"  {p}")
    if len(problems) > args.show:
        print(f"  ... and {len(problems) - args.show} more")

    if problems:
        print(f"\n{len(problems)} problem(s).")
        return 1

    samples = load_pairs(args.path)     # prints what it drops and why
    n_vuln = sum(s.label == 1 for s in samples)
    if n_vuln * 2 != len(samples):
        print(f"\nunbalanced: {n_vuln} vulnerable of {len(samples)} samples")
        return 1
    print("\nThe file matches the contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
