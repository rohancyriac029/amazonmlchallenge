"""Blocking benchmark on a sample of S1 queries (dev folds only)."""
import sys, time, numpy as np, pandas as pd
import config, blocking
from data_loader import read_ground_truth
from candidates import label_pairs
from prep import fold_of
df = pd.read_parquet(config.work("train_v1.parquet")); gt = read_ground_truth()
rng = np.random.default_rng(0)
s1 = np.flatnonzero((df.src == 1).values)
s1 = s1[[fold_of(x) != 0 for x in df.entity_id.values[s1]]]
qsel = rng.choice(s1, int(sys.argv[1]) if len(sys.argv) > 1 else 40000, replace=False)
q_mask = np.zeros(len(df), bool); q_mask[qsel] = True
max_df = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
t = time.time(); pairs = blocking.generate_candidates(df, q_mask, max_df=max_df); print(f"time {time.time()-t:.0f}s")
pairs["y"] = label_pairs(df, pairs, gt)
ids = df.entity_id.values
tot = sum(len(gt[i]) for i in ids[qsel])
cty = df.country.values[pairs.q.values]
print(f"UNION recall={pairs.y.sum()/tot:.5f} per_q={len(pairs)/len(qsel):.1f}")
for c in np.unique(cty):
    m = cty == c; qn = len(np.intersect1d(qsel, np.flatnonzero(df.country.values == c)))
    tc = sum(len(gt[i]) for i in ids[qsel][df.country.values[qsel] == c])
    print(f"  {c}: recall={pairs.y[m].sum()/tc:.5f} per_q={m.sum()/qn:.1f}")
sc = [c for c in pairs.columns if c.startswith("s_")]
for c in sc:
    r = pairs.groupby("q")[c].rank(ascending=False, method="first")
    print(c, {k: round(pairs.y[r <= k].sum() / tot, 4) for k in (5, 10, 20, 30)})
# learned-free rank fusion: max over channels of reciprocal rank
R = np.zeros(len(pairs))
for c in sc:
    R = np.maximum(R, 1 / pairs.groupby("q")[c].rank(ascending=False, method="first").values)
pairs["rrf"] = R + 0.001 * pairs.s_combo.values
r = pairs.groupby("q").rrf.rank(ascending=False, method="first")
print("fused max-RR", {k: round(pairs.y[r <= k].sum() / tot, 4) for k in (10, 15, 20, 25, 30, 40, 60)})
pairs.to_parquet(config.work("bench_pairs.parquet"))
