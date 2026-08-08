"""Loading the pair dataset.

One JSONL line is a minimal pair: the same function before and after its fix. Each
line becomes two Samples sharing an `idx`, the key every pairwise metric groups on.
"""

import json
import random
import re
from dataclasses import dataclass, field

# Some bodies keep the language tag of the markdown fence they were extracted from.
_FENCE_TAG = re.compile(r"^(c|cpp|cxx|h)\n", re.IGNORECASE)


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


def _clean_body(body):
    return _FENCE_TAG.sub("", body, count=1).strip()


def load_pairs(path, commit_mode="parent-of-fix"):
    """Read the dataset and explode every pair into its two halves.

    commit_mode "vul-intro" puts the vulnerable half on the introducing commit
    instead of the parent of the fix. Ablation only.
    """
    with open(path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]

    # A repeated idx would put four samples under one pair_id, which no pairwise
    # metric can group.
    seen = {}
    for d in raw:
        seen.setdefault(d["idx"], d)
    rows = list(seen.values())

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
            func_body=_clean_body(d["vulnerable_function_body"]), label=1, **common,
        ))
        samples.append(Sample(
            sample_id=f"{idx}-benign", commit=d["vulnerability_fixing_commit_id"],
            func_body=_clean_body(d["non_vulnerable_function_body"]), label=0, **common,
        ))

    n_vuln = sum(s.label == 1 for s in samples)
    print(f"[data] {len(raw)} lines -> {len(rows)} pairs after idx dedup "
          f"-> {len(samples)} samples ({n_vuln} vuln / {len(samples) - n_vuln} benign)",
          flush=True)
    return samples


def stratified_subsample(samples, n_pairs=120, seed=0):
    """Draw about n_pairs pairs, stratified by (project, primary CWE).

    Round-robins across strata so the sample is not dominated by whichever project
    contributes the most pairs.
    """
    pairs = {}
    for s in samples:
        pairs.setdefault(s.pair_id, []).append(s)

    def stratum(pid):
        s = pairs[pid][0]
        return (s.project, s.cwe[0] if s.cwe else "none")

    groups = {}
    for pid in sorted(pairs):
        groups.setdefault(stratum(pid), []).append(pid)

    rng = random.Random(seed)
    strata = sorted(groups)
    rng.shuffle(strata)
    for k in strata:
        rng.shuffle(groups[k])

    selected = []
    while len(selected) < n_pairs and any(groups[k] for k in strata):
        for k in strata:
            if groups[k]:
                selected.append(groups[k].pop())
            if len(selected) >= n_pairs:
                break

    out = []
    for pid in selected:
        out.extend(pairs[pid])
    print(f"[data] subsample: {len(selected)} pairs over {len(strata)} strata "
          f"(seed {seed}) -> {len(out)} samples", flush=True)
    return out
