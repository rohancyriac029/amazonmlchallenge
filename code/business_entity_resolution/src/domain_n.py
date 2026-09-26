"""E1 acceptance check: does adding the house-number relation features raise train-vs-test domain AUC?"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import config
import exp
import final2

spec = final2.s2_spec("inv3n")                      # v3 stage-2 columns + A8 + N
Xtr, mtr, cols = exp.load("train_feat", spec["base_cols"], spec["extra"])
Xte, mte, _ = exp.load("test_feat", spec["base_cols"], spec["extra"])
ctr = pd.read_parquet(config.work("train_v1.parquet"), columns=["country"]).country.values[mtr.q.values]
cte = pd.read_parquet(config.work("test_v1.parquet"), columns=["country"]).country.values[mte.q.values]
import features_num
num = set(features_num.N_FEATS)
rng = np.random.default_rng(0)
for country in ("US", "India"):
    ia = rng.choice(np.flatnonzero(ctr == country), 1_000_000, replace=False)
    ib = rng.choice(np.flatnonzero(cte == country), 1_000_000, replace=False)
    X = np.vstack([Xtr[ia], Xte[ib]]); y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]; n = int(0.8 * len(y))
    for name, keep in (("v3 stage-2 (without N)", [i for i, c in enumerate(cols) if c not in num]),
                       ("v3 + N (house-number relations)", list(range(len(cols)))),
                       ("N features alone", [i for i, c in enumerate(cols) if c in num])):
        m = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.1, verbose=-1, num_threads=8),
                      lgb.Dataset(X[:n][:, keep], y[:n]), 150)
        auc = roc_auc_score(y[n:], m.predict(X[n:][:, keep]))
        imp = pd.Series(m.feature_importance("gain"), index=[cols[i] for i in keep]).sort_values(ascending=False)
        print(f"{country:6s} {name:34s} AUC={auc:.4f} top={list(imp.index[:4])}", flush=True)
