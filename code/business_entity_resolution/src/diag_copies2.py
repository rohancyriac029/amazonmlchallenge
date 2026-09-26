"""Train: how often is a strong near-copy a true match; unowned near-copies per S1; by #near-copies."""
import numpy as np, pandas as pd
import config
from data_loader import read_ground_truth
F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "n_tset", "a_tset"])
df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id"])
gt = read_ground_truth()
owned = {o for v in gt.values() for o in v}
strong = (F.n_tset.values >= 90) & (F.a_tset.values >= 85)
S = F[strong].copy()
S["owned"] = [x in owned for x in df.entity_id.values[S.p.values]]
print(f"strong near-copy candidates: {len(S):,}; true match {S.y.mean():.3f}; owned by another S1 {np.mean(S.owned & (S.y == 0)):.3f}; unowned {np.mean(~S.owned):.3f}")
per = S.groupby("q").agg(n=("y", "size"), tp=("y", "sum"), unowned=("owned", lambda s: int((~s).sum())))
print(f"per S1 with >=1 strong: strong {per.n.mean():.2f}, true {per.tp.mean():.2f}, unowned {per.unowned.mean():.2f}")
per["bucket"] = per.n.clip(upper=7)
print(per.groupby("bucket").agg(S1=("n", "size"), true_per=("tp", "mean"), unowned_per=("unowned", "mean")).round(3).to_string())
