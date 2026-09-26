"""Step 2: candidate generation for a split -> work/{split}_cands.parquet (+ labels on train)."""
import sys
import time

import numpy as np
import pandas as pd

import config
import blocking
from data_loader import read_ground_truth
from prep import fold_of


def label_pairs(df, pairs, gt):
    """y=1 iff the pool record's true owner is the query S1 (vectorised)."""
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owner_s1 = [s for s, v in gt.items() for _ in v]
    owned = [o for v in gt.values() for o in v]
    owner_row = np.full(len(df), -1, dtype=np.int64)
    owner_row[pos.reindex(owned).values] = pos.reindex(owner_s1).values
    return (owner_row[pairs.p.values] == pairs.q.values).astype(np.int8)


def fold_array(df):
    """fold id per row of df (S1 rows only meaningful)."""
    f = np.full(len(df), -1, dtype=np.int8)
    is1 = np.flatnonzero((df.src == 1).values)
    f[is1] = [fold_of(x) for x in df.entity_id.values[is1]]
    return f


def recall_report(df, pairs, gt, tag=""):
    ids = df.entity_id.values
    tot = sum(len(v) for v in gt.values())
    hit = int(pairs.y.sum())
    nq = int((df.src == 1).sum())
    npool = int((df.src != 1).sum())
    print(f"{tag} pairs={len(pairs):,} per_q={len(pairs) / nq:.1f} recall={hit / tot:.5f} "
          f"reduction_ratio={1 - len(pairs) / (nq * npool):.8f}")
    for ch in [c for c in pairs.columns if c.startswith("s_")]:
        r = pairs.groupby("q")[ch].rank(ascending=False, method="first")
        print(f"   {ch}:", {k: round(pairs.y[r <= k].sum() / tot, 5) for k in (5, 10, 20, 30)})
    from train import fused_rank
    r = fused_rank(pairs)
    print("   fused:", {k: round(pairs.y[r <= k].sum() / tot, 5) for k in (10, 15, 20, 25, 30, 40, 60)})


def run(split, **kw):
    t0 = time.time()
    df = pd.read_parquet(config.work(f"{split}_v1.parquet"))
    q_mask = (df.src == 1).values
    pairs = blocking.generate_candidates(df, q_mask, **kw)
    print(f"candidates done {time.time() - t0:.0f}s")
    if split.startswith("train"):
        gt = read_ground_truth()
        pairs["y"] = label_pairs(df, pairs, gt)
        pairs["fold"] = fold_array(df)[pairs.q.values]
        recall_report(df, pairs, gt, split)
    pairs.to_parquet(config.work(f"{split}_cands.parquet"), index=False)
    return pairs


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "train")
