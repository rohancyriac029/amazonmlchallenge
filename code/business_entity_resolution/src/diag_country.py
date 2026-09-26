"""Label-free per-country confidence profile: test predictions vs. holdout predictions."""
import numpy as np, pandas as pd
import config, decision, train
def profile(meta, df, tag):
    cty = df.country.values[meta.q.values]
    g = pd.DataFrame({"q": meta.q.values, "prob": meta.prob.values, "cty": cty})
    mx = g.groupby("q").agg(mx=("prob", "max"), cty=("cty", "first"),
                            n_mid=("prob", lambda s: int(((s > 0.1) & (s < 0.9)).sum())),
                            esum=("prob", "sum"))
    for c, s in mx.groupby("cty"):
        print(f"{tag:8s} {c:7s} n={len(s):>7,} best>=0.9: {np.mean(s.mx >= 0.9):.3f}  best<0.6: {np.mean(s.mx < 0.6):.3f}  "
              f"uncertain(0.1-0.9) per S1: {s.n_mid.mean():.2f}  expected matches: {s.esum.mean():.2f}")
# holdout (fold 0) with stack1 OOF probs
df = pd.read_parquet(config.work("train_v1.parquet"), columns=["country"])
m = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "fold"])
m["prob"] = np.load(config.work("oof_stack1.npy"))
profile(m[m.fold == 0], df, "holdout")
dt = pd.read_parquet(config.work("test_v1.parquet"), columns=["country"])
mt = pd.read_parquet(config.work("test_feat.parquet"), columns=["q", "p"])
mt["prob"] = np.load(config.work("test_prob.npy"))
profile(mt, dt, "test")
