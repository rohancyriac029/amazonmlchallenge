"""R4: print samples of residual house-number errors of the final model (dev folds only)."""
import numpy as np
import pandas as pd

import config

F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold", "n_tset", "a_tset", "a_first_num_eq", "a_nn_q", "a_nn_p"])
N = pd.read_parquet(config.work("train_feat_N.parquet"), columns=["num_logdiff", "num_neighbour"])
F = pd.concat([F, N], axis=1)
F["prob"] = np.load(config.work("oof_s3inv3n.npy"))
F = F[F.fold != 0]
best = F.groupby("p").prob.transform("max")
acc = (F.prob >= 0.9) & (F.prob >= best)
h = (F.a_nn_q > 0) & (F.a_nn_p > 0) & (F.a_first_num_eq == 0) & (F.n_tset >= 90) & (F.num_logdiff <= np.log1p(12))
df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "business_name", "business_address"])
rng = np.random.default_rng(3)
for name, m in (("TRUE MATCH rejected (FN), small |delta|", h & (F.y == 1) & ~acc), ("DISTRACTOR accepted (FP), small |delta|", h & (F.y == 0) & acc)):
    X = F[m]
    idx = rng.choice(len(X), min(40, len(X)), replace=False)
    print(f"\n===== {name}: {m.sum():,} (showing {len(idx)})")
    for r in X.iloc[idx].itertuples():
        a, b = df.iloc[r.q], df.iloc[r.p]
        print(f"  p={r.prob:.2f} | S1: {a.business_name} | {a.business_address}\n            OT: {b.business_name} | {b.business_address}")
