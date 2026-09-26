"""Domain AUC (train vs test) for candidate feature subsets: lower = less covariate shift."""
import numpy as np, pandas as pd, lightgbm as lgb
import config, train
from sklearn.metrics import roc_auc_score
ALL = [c for c in train.feature_columns("train") if c not in train.META]
SIZE = [c for c in ALL if "_f_" in c]
COMP = [c for c in ALL if c.endswith("_gap_p") or c.endswith("_rank_p") or c in ("n_cand_p",)]
S1 = ["s1_score", "s1_rank"]
SETS = {"full": [], "-size": SIZE, "-size-comp": SIZE + COMP, "-size-comp-s1": SIZE + COMP + S1}
rng = np.random.default_rng(0)
def sample(split, n, country):
    X = train.load_matrix(split, ALL)
    m = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=["q"])
    cty = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["country"]).country.values[m.q.values]
    idx = rng.choice(np.flatnonzero(cty == country), n, replace=False)
    return X[idx]
for country in ("US", "India"):
    Xa, Xb = sample("train", 1_000_000, country), sample("test", 1_000_000, country)
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]; n = int(len(y) * 0.8)
    for name, drop in SETS.items():
        keep = [i for i, c in enumerate(ALL) if c not in drop]
        m = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.1, verbose=-1, num_threads=8),
                      lgb.Dataset(X[:n][:, keep], y[:n]), 150)
        auc = roc_auc_score(y[n:], m.predict(X[n:][:, keep]))
        imp = pd.Series(m.feature_importance("gain"), index=[ALL[i] for i in keep]).sort_values(ascending=False)
        print(f"{country:6s} {name:15s} n_feat={len(keep):3d} AUC={auc:.4f}  top: {list(imp.index[:4])}", flush=True)
