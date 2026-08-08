"""Loading the pair dataset.

One JSONL line is a minimal pair: the same function before and after its fix. Each
line becomes two Samples sharing an `idx`, the key every pairwise metric groups on.
"""

import json
from dataclasses import dataclass, field


@dataclass
class Sample:
    sample_id: str
    pair_id: int            # = idx
    cve: str
    cwe: list = field(default_factory=list)
    project: str = ""
    project_url: str = ""
    file_name: str = ""     # repo-relative
    commit: str = ""        # the commit this half of the pair lives at
    func_body: str = ""
    label: int = 0          # 1 vulnerable, 0 benign


def load_pairs(path, commit_mode="parent-of-fix"):
    """Read the dataset and explode every pair into its two halves.

    commit_mode "vul-intro" puts the vulnerable half on the introducing commit
    instead of the parent of the fix. Ablation only.
    """
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    samples = []
    for d in rows:
        idx = d["idx"]
        common = dict(
            pair_id=idx, cve=d["cve"], cwe=d["cwe"],
            project=d["project"], project_url=d["project_url"],
            file_name=d["file_name"],
        )
        vuln_commit = (
            d["vulnerability_introducing_commit_id"] if commit_mode == "vul-intro"
            else d["vulnerable_commit_id"]
        )
        samples.append(Sample(
            sample_id=f"{idx}-vuln", commit=vuln_commit,
            func_body=d["vulnerable_function_body"], label=1, **common,
        ))
        samples.append(Sample(
            sample_id=f"{idx}-benign", commit=d["vulnerability_fixing_commit_id"],
            func_body=d["non_vulnerable_function_body"], label=0, **common,
        ))

    n_vuln = sum(s.label == 1 for s in samples)
    print(f"[data] {len(rows)} pairs -> {len(samples)} samples "
          f"({n_vuln} vuln / {len(samples) - n_vuln} benign)", flush=True)
    return samples
