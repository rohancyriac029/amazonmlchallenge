"""Label-free: near-copy candidates per S1 (name AND address nearly identical), train vs test."""
import numpy as np, pandas as pd
import config
from data_loader import read_ground_truth
cols = ["q", "p", "n_tset", "a_tset", "p_a_empty"]
for split in ("train", "test"):
    F = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=cols)
    cty = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["country"]).country.values[F.q.values]
    strong = (F.n_tset.values >= 90) & (F.a_tset.values >= 85)
    name_only = (F.n_tset.values >= 90) & (F.p_a_empty.values == 1)
    g = pd.DataFrame({"q": F.q.values, "c": cty, "strong": strong, "name_only": name_only})
    per = g.groupby("q").agg(c=("c", "first"), strong=("strong", "sum"), name_only=("name_only", "sum"))
    for c, s in per.groupby("c"):
        print(f"{split:5s} {c:7s} strong near-copies per S1 {s.strong.mean():.3f}  (>=5: {np.mean(s.strong >= 5):.3f})  name-only strong {s.name_only.mean():.3f}")
# for reference: true matches per S1 in train, and how many strong candidates are true
gt = read_ground_truth()
