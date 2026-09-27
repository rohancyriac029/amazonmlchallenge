"""Label-free look at France (unseen in training, 15% of test S1): confidence profile vs US/India and
samples of uncertain French S1 with their top candidates, raw and normalised text side by side."""
import numpy as np
import pandas as pd

import config

TAG = "s3inv3n_ce"

if __name__ == "__main__":
    F = pd.read_parquet(config.work("test_feat.parquet"), columns=["q", "p"])
    F["prob"] = np.load(config.work(f"test_prob_{TAG}.npy"))
    df = pd.read_parquet(config.work("test_v1.parquet"),
                         columns=["entity_id", "country", "business_name", "business_address", "n_core", "a_norm"])
    F["cty"] = df.country.values[F.q.values]
    best = F.sort_values("prob", ascending=False).drop_duplicates("q")
    print("best-candidate probability profile per country (share of S1):")
    bands = pd.cut(best.prob, [0, 0.1, 0.5, 0.9, 0.99, 1.0])
    print(pd.crosstab(best.cty, bands, normalize="index").round(3).to_string())
    n_conf = F[F.prob >= 0.9].groupby("q").size().reindex(best.q, fill_value=0)
    print("\nmean candidates >= 0.9 per S1:", n_conf.groupby(best.cty.values).mean().round(3).to_dict())
    rng = np.random.default_rng(0)
    fr = best[(best.cty == "France") & (best.prob > 0.3) & (best.prob < 0.9)]
    for q in rng.choice(fr.q.values, 12, replace=False):
        c = F[F.q == q].nlargest(3, "prob")
        print(f"\nS1: {df.business_name[q]} | {df.business_address[q]}   [norm: {df.n_core[q]} | {df.a_norm[q]}]")
        for r in c.itertuples():
            print(f"  {r.prob:.2f}  {df.business_name[r.p]} | {df.business_address[r.p]}   [norm: {df.n_core[r.p]} | {df.a_norm[r.p]}]")
