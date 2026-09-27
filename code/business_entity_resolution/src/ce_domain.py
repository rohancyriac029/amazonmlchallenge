"""CE acceptance check: does the cross-encoder logit add train-vs-test shift beyond stage 2?

Domain classifier (train pairs vs test pairs, same selection rule, CE-scored pairs only) on
[logit p2] vs [logit p2, ce_logit], per country.  The train side is split into the dev folds
(scored by one half-model, what stage 3 is trained on) and holdout fold 0 (mean of both
half-models, exactly like test).  Pre-registered: the AUC rise from adding the CE is <= 0.01.

  python ce_domain.py [ref_feature ...]   ce_bf16 = CE rounded to the bfloat16 grid (numeric-artefact test);
                                          ref features show the rise a trusted pair feature produces
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config


def side(split):
    P = pd.read_parquet(config.work(f"ce_pairs_{split}.parquet"), columns=["row", "p2", "q_id"] + (["fold"] if split == "train" else []))
    P["ce"] = np.load(config.work(f"ce_logit_{split}.npy"))
    P = P[P.ce.notna()]
    cty = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["entity_id", "country"]).set_index("entity_id").country
    P["country"] = cty.reindex(P.q_id.values).values
    return P


def auc(A, B, cols, rng):
    X = np.r_[A[cols].values, B[cols].values].astype(np.float32)
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    half = rng.random(len(y)) < 0.5
    s = np.empty(len(y))
    for tr in (half, ~half):
        m = lgb.train({"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "verbose": -1,
                       "num_threads": config.N_JOBS}, lgb.Dataset(X[tr], y[tr]), 100)
        s[~tr] = m.predict(X[~tr])
    return roc_auc_score(y, s)


def to_bf16(x):
    """Round float32 to the nearest bfloat16 value (round-to-nearest-even)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


if __name__ == "__main__":
    import sys
    tr, te = side("train"), side("test")
    ref = sys.argv[1:]                      # optional trusted stage-2 features as a reference for "normal" shift
    for split, D in (("train", tr), ("test", te)):
        D["lp2"] = np.log(np.clip(D.p2, 1e-6, 1 - 1e-6) / (1 - np.clip(D.p2, 1e-6, 1 - 1e-6)))
        D["ce_bf16"] = to_bf16(D.ce.values)
        if ref:
            F = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=ref)
            for c in ref:
                D[c] = F[c].values[D.row.values]
    for name, m in (("dev folds", tr.fold != 0), ("fold 0", tr.fold == 0)):
        print(f"share of CE values on the bf16 grid: {name} {np.mean(tr.ce.values[m] == tr.ce_bf16.values[m]):.3f}", end=" | ")
    print(f"test {np.mean(te.ce.values == te.ce_bf16.values):.3f}")
    rng = np.random.default_rng(0)
    n = 500_000
    for country in ("US", "India"):
        B = te[te.country == country]
        B = B.iloc[rng.choice(len(B), min(n, len(B)), replace=False)]
        for name, A in (("dev folds", tr[(tr.country == country) & (tr.fold != 0)]),
                        ("fold 0", tr[(tr.country == country) & (tr.fold == 0)])):
            A = A.iloc[rng.choice(len(A), min(n, len(A)), replace=False)]
            a0 = auc(A, B, ["lp2"], rng)
            res = {c: auc(A, B, ["lp2", c], rng) - a0 for c in ["ce", "ce_bf16"] + ref}
            print(f"{country:5s} {name:9s} vs test: domain AUC p2 {a0:.4f}; rise when adding " +
                  ", ".join(f"{c} {v:+.4f}" for c, v in res.items()))
