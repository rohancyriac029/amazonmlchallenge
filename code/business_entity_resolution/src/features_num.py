"""E1: house-number relation features (pair-intrinsic, no dataset statistics).

Measured on training near-duplicates whose house number differs from the S1's:
unowned distractors are neighbouring premises (median |delta| 7, 69% within 12,
25% same parity), true matches differ by corruption (median |delta| 301, 18%
truncation, 57% same parity).  The existing numeric similarity is character-based
and cannot express this.  Every feature is three-state: agree/relation holds (1),
conflict/does not hold (0), unknown because a side has no number (-1).

  python features_num.py <split> <feat_table>   -> work/<feat_table>_N.parquet (row-aligned)
"""
import math
import sys
import time

import numpy as np
import pandas as pd

import config

_N = {}
N_FEATS = ["num_state", "num_logdiff", "num_same_parity", "num_same_len", "num_one_digit_sub",
           "num_transposition", "num_prefix", "num_neighbour", "num_min_logdiff_any"]


def _rel(a, b, words_a, words_b, nums_b):
    if not a or not b:
        return [-1.0] * 9
    ia, ib = int(a), int(b)
    agree = float(a == b)
    d = abs(ia - ib)
    same_len = float(len(a) == len(b))
    one_sub = float(len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1)
    transp = float(a != b and sorted(a) == sorted(b))
    prefix = float(a != b and (a.startswith(b) or b.startswith(a)))
    wa, wb = set(words_a.split()), set(words_b.split())
    street_overlap = len(wa & wb) / max(len(wa | wb), 1)
    neighbour = float(a != b and d <= 12 and street_overlap >= 0.5)
    others = [int(x) for x in nums_b.split() if x.isdigit() and len(x) <= 12]
    min_any = min((abs(ia - x) for x in others), default=d)
    return [agree, math.log1p(d), float(ia % 2 == ib % 2), same_len, one_sub, transp, prefix, neighbour,
            math.log1p(min_any)]


def _block(bounds):
    s, e = bounds
    q, p = _N["q"][s:e], _N["p"][s:e]
    fn, wd, nm = _N["fnum"], _N["words"], _N["nums"]
    out = np.empty((e - s, len(N_FEATS)), dtype=np.float32)
    for i, (qi, pi) in enumerate(zip(q, p)):
        out[i] = _rel(fn[qi], fn[pi], wd[qi], wd[pi], nm[pi])
    return out


def build(split, table, chunk=50000):
    t0 = time.time()
    df = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["a_first_num", "a_words", "a_nums"])
    F = pd.read_parquet(config.work(f"{table}.parquet"), columns=["q", "p"])
    shared = {"q": F.q.values, "p": F.p.values, "fnum": df.a_first_num.tolist(),
              "words": df.a_words.tolist(), "nums": df.a_nums.tolist()}
    bounds = [(s, min(s + chunk, len(F))) for s in range(0, len(F), chunk)]
    X = np.vstack(config.parallel_map(_block, bounds, "_N", shared))
    pd.DataFrame(X, columns=N_FEATS).to_parquet(config.work(f"{table}_N.parquet"), index=False)
    print(f"{table}_N {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
