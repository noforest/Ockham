"""The detection prompts, read from prompts.toml at the repo root: one entry, one prompt."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "prompts.toml"
_COUNT = {1: "one line", 2: "two lines", 3: "three lines", 4: "four lines"}


@dataclass
class Prompt:
    name: str
    text: str
    verdict: str            # "first" or "last": which VERDICT: line the parser reads
    ask_cwe: bool
    max_tokens: int
    sha: str


def _config():
    with open(PATH, "rb") as f:
        return tomllib.load(f)


def names():
    return list(_config()["prompts"])


def load(name):
    """The entry rendered with [vars]; the RETURN_OPT header counts its own lines."""
    cfg = _config()
    entry, v = cfg["prompts"][name], cfg["vars"]
    lines = entry.get("lines", [])
    if "text" in entry:
        text = entry["text"]
    else:
        spec = [cfg["lines"][k].format_map(v) for k in lines]
        ret = f"Return exactly {_COUNT[len(spec)]} and nothing else:\n" + "\n".join(spec)
        text = cfg[entry.get("template", "template")].format_map({**v, "RETURN_OPT": ret})
    text = text.strip()
    return Prompt(name, text, entry.get("verdict", "first"), "cwe" in lines,
                  entry["max_tokens"], hashlib.sha256(text.encode()).hexdigest()[:12])
