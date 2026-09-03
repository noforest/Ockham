"""Blocking checks on the pair file: a pair that is not a pair must never reach a cell."""

import sys
from pathlib import Path

OCKHAM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OCKHAM))

from ockham.data import load_pairs
from ockham.run import DATA          # the one path the runs actually use


def _pairs():
    out = {}
    for s in load_pairs(DATA):
        out.setdefault(s.pair_id, []).append(s)
    return out


def test_every_pair_has_two_distinct_halves():
    """The halves must differ by more than whitespace, or no model can score P-C on it."""
    same = [pid for pid, ss in _pairs().items()
            if len({" ".join(s.func_body.split()) for s in ss}) == 1]
    assert not same, f"{len(same)} pairs with identical halves survived the load: {same[:5]}"


def test_every_pair_is_one_vulnerable_and_one_benign():
    """pAcc groups on pair_id; anything but 1+0 matches no P-C/P-V/P-B/P-R bucket."""
    bad = [pid for pid, ss in _pairs().items() if sorted(s.label for s in ss) != [0, 1]]
    assert not bad, f"{len(bad)} malformed pairs: {bad[:5]}"


def test_the_drop_is_the_known_defect_and_not_a_regression():
    """~18 % of JitVul goes; a number far off means the loader changed, not the data."""
    n = len(_pairs())
    assert 630 <= n <= 680, f"{n} pairs kept, expected ~654 of 800"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
