"""Loading the pair file: one line becomes two Samples sharing an idx, unless they match."""

import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# Some bodies keep the language tag of the markdown fence they were extracted from.
_FENCE_TAG = re.compile(r"^(c|cpp|cxx|h)\n", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

GIT_TIMEOUT_S = 300


@dataclass
class Sample:
    sample_id: str
    pair_id: int            # = idx
    cve: str
    cwe: list = field(default_factory=list)
    project: str = ""
    project_url: str = ""
    file_name: str = ""
    commit: str = ""
    func_body: str = ""
    label: int = 0          # 1 vulnerable, 0 benign


def _clean_body(body):
    return _FENCE_TAG.sub("", body, count=1).strip()


def _same_code(a, b):
    """True when two halves differ only in whitespace, so no patch separates them."""
    return _WHITESPACE.sub(" ", a).strip() == _WHITESPACE.sub(" ", b).strip()


def load_pairs(path, commit_mode="parent-of-fix"):
    """Explode every pair into its two halves, at the commit the mode selects."""
    with open(path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]

    # A repeated idx would put four samples under one pair_id.
    seen = {}
    for d in raw:
        seen.setdefault(d["idx"], d)
    rows = list(seen.values())

    samples = []
    n_identical = 0
    for d in rows:
        idx = d["idx"]
        vuln_body = _clean_body(d["vulnerable_function_body"])
        benign_body = _clean_body(d["non_vulnerable_function_body"])

        # 146 of 800 pairs hold the same code on both sides: forced P-V/P-B, not a hard case.
        if _same_code(vuln_body, benign_body):
            n_identical += 1
            continue

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
            func_body=vuln_body, label=1, **common,
        ))
        samples.append(Sample(
            sample_id=f"{idx}-benign", commit=d["vulnerability_fixing_commit_id"],
            func_body=benign_body, label=0, **common,
        ))

    n_vuln = sum(s.label == 1 for s in samples)
    print(f"[data] {len(raw)} lines -> {len(rows)} pairs after idx dedup "
          f"-> {len(rows) - n_identical} pairs after dropping {n_identical} with identical "
          f"halves -> {len(samples)} samples "
          f"({n_vuln} vuln / {len(samples) - n_vuln} benign)", flush=True)
    return samples


def stratified_subsample(samples, n_pairs=120, seed=0):
    """Draw about n_pairs pairs, round-robin over the (project, primary cwe) strata."""
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


def _git(args, cwd):
    """Run git, reporting a timeout as a failed call rather than hanging the run."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 1, "", "timeout")


def bare_clone(project_url, project, workspace):
    """Clone the repository once, blobless, and reuse it for every commit."""
    repo_dir = (Path(workspace) / "repos" / f"{project}.git").resolve()
    if not repo_dir.exists():
        print(f"[data] cloning {project} ({project_url})...", flush=True)
        t0 = time.time()
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--bare", "--filter=blob:none", project_url, str(repo_dir)], cwd=None)
        print(f"[data] clone {project}: {'ok' if repo_dir.exists() else 'FAILED'} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return repo_dir if repo_dir.exists() else None


def add_worktree(repo_dir, commit, project, workspace):
    """Check the repository out at one commit, in its own detached worktree."""
    wt_dir = (Path(workspace) / "worktrees" / f"{project}-{commit[:12]}").resolve()
    if wt_dir.exists():
        return wt_dir
    print(f"[data] fetching {project}@{commit[:12]}...", flush=True)
    t0 = time.time()
    _git(["fetch", "origin", commit], cwd=repo_dir)
    r = _git(["worktree", "add", "--detach", str(wt_dir), commit], cwd=repo_dir)
    ok = r.returncode == 0
    print(f"[data] worktree {project}@{commit[:12]}: {'ok' if ok else 'FAILED'} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return wt_dir if ok else None


def checkout(sample, workspace):
    """The worktree holding this sample's commit, or None if it could not be built."""
    repo_dir = bare_clone(sample.project_url, sample.project, workspace)
    if repo_dir is None:
        return None
    return add_worktree(repo_dir, sample.commit, sample.project, workspace)
