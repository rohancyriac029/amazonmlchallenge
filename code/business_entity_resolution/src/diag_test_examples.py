"""Inspect test S1s with many high-probability candidates (label-free) and per-source counts vs train truth."""
import numpy as np, pandas as pd
import config
from data_loader import read_ground_truth
meta = pd.read_parquet(config.work("test_feat.parquet"), columns=["q", "p", "n_tset", "a_tset"])
meta["prob"] = np.load(config.work("test_prob.npy"))
df = pd.read_parquet(config.work("test_v1.parquet"), columns=["entity_id", "business_name", "business_address", "country", "src"])
hi = meta[meta.prob >= 0.5]
cnt = hi.groupby("q").size()
src = df.src.values[hi.p.values]
per = pd.DataFrame({"q": hi.q.values, "s2": src == 2, "s3": src == 3}).groupby("q").sum()
print("test: accepted-ish (p>=0.5) per S1 by source: S2 mean %.2f max %d | S3 mean %.2f max %d" % (per.s2.mean(), per.s2.max(), per.s3.mean(), per.s3.max()))
print("test: share S1 with >5 S2 or >6 S3 (above train max):", float(((per.s2 > 5) | (per.s3 > 6)).mean()))
print("test: dist of #p>=0.5 per S1:", cnt.value_counts().sort_index().head(14).to_dict())
gt = read_ground_truth()
sz = pd.Series([len(v) for v in gt.values()])
print("train truth: dist of #matches per S1:", sz.value_counts().sort_index().to_dict())
rng = np.random.default_rng(1)
many = cnt[cnt >= 8].index.values
for q in rng.choice(many, 4, replace=False):
    r = df.iloc[q]
    print(f"\n### {r.entity_id} | {r.business_name} | {r.business_address} | {r.country}")
    c = meta[meta.q == q].sort_values("prob", ascending=False).head(12)
    for x in c.itertuples():
        o = df.iloc[x.p]
        print(f"   p={x.prob:.3f} {o.entity_id} | {o.business_name} | {o.business_address}")
