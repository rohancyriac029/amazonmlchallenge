"""Domain classifier: train vs test candidate pairs (label-free covariate-shift diagnostic)."""
import numpy as np, pandas as pd, lightgbm as lgb
import config, train
from sklearn.metrics import roc_auc_score
cols = [c for c in train.feature_columns("train") if c not in train.META]
rng = np.random.default_rng(0)
import sys
TRAIN_TABLE = sys.argv[1] if len(sys.argv) > 1 else "train"   # "train" or "train_univ"
def sample(split, n, country=None):
    X = train.load_matrix(split, cols)
    m = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=["q"])
    base = split.split("_")[0]
    cty = pd.read_parquet(config.work(f"{base}_v1.parquet"), columns=["country"]).country.values[m.q.values]
    idx = np.flatnonzero(cty == country) if country else np.arange(len(X))
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    return X[idx], cty[idx]
for country in ("US", "India"):
    Xa, _ = sample(TRAIN_TABLE, 1_500_000, country)
    Xb, _ = sample("test", 1_500_000, country)
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    n = int(len(y) * 0.8)
    m = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.1, verbose=-1, num_threads=8),
                  lgb.Dataset(X[:n], y[:n], feature_name=cols), 200)
    auc = roc_auc_score(y[n:], m.predict(X[n:]))
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print(f"\n{country}: train-vs-test AUC = {auc:.4f}; top shifting features:")
    print((imp / imp.sum()).head(12).round(3).to_string())
    for c in imp.index[:6]:
        j = cols.index(c)
        print(f"   {c:22s} train mean {np.nanmean(Xa[:, j]):9.3f}  test mean {np.nanmean(Xb[:, j]):9.3f}")
