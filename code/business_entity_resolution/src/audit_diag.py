"""Cheap diagnostics requested by the generalisation audit (no retraining, no holdout labels).

  python audit_diag.py ids      record IDs: any ordering / correlation leak?
  python audit_diag.py domain   E2: train-vs-test domain AUC of v2 stage-3 feature sets
  python audit_diag.py pd       E4: partial dependence of v2 stage 2 on key evidence features
  python audit_diag.py calib    E5: calibration (ECE) of v2 OOF probabilities by stratum (dev folds only)
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import exp
import stack

AGG = ["prob_sum_q", "n_conf_q", "n_conf90_q", "n_conf_p"]


def within_s1(cols):
    return [c for c in cols if c.endswith(("_rank_q", "_max_q", "_gap_q")) or c in ("n_cand_q",)]


def ids():
    from data_loader import read_ground_truth
    gt = read_ground_truth()
    num = lambda x: int(x.split("-")[1])
    a, b = [], []
    for s, v in list(gt.items())[:500000]:
        for o in v:
            a.append(num(s)); b.append(num(o))
    a, b = np.array(a), np.array(b)
    print(f"true pairs: Spearman(S1 id, matched id) = {pd.Series(a).corr(pd.Series(b), method='spearman'):+.4f}")
    rng = np.random.default_rng(0)
    print(f"random pairs baseline: {pd.Series(a).corr(pd.Series(rng.permutation(b)), method='spearman'):+.4f}")
    sz = np.array([len(v) for v in gt.values()]); ids_ = np.array([num(s) for s in gt])
    print(f"Spearman(S1 id, #matches) = {pd.Series(ids_).corr(pd.Series(sz), method='spearman'):+.4f}")


def domain():
    sp = json.load(open(config.work("spec_s3inv2.json")))
    spec = sp["spec"]
    # train: stage-3 inputs exactly as used for training (OOF stage-2 probabilities)
    Str = pd.read_parquet(config.work("train_stack_inv2.parquet"))
    Xtr, mtr, cols = exp.load("train_feat", spec["base_cols"], spec["extra"])
    ctr = pd.read_parquet(config.work("train_v1.parquet"), columns=["country"]).country.values[mtr.q.values]
    # test: stage-2 full-model probabilities -> stack features
    Xte, mte, _ = exp.load("test_feat", spec["base_cols"], spec["extra"])
    p2 = lgb.Booster(model_file=config.work("stage2_s3inv2.txt")).predict(Xte, num_threads=config.N_JOBS)
    dte = pd.read_parquet(config.work("test_v1.parquet"), columns=["country", "n_core", "a_norm", "a_first_num", "src"])
    Ste = stack.stack_features(dte, mte, p2.astype(np.float32))
    cte = dte.country.values[mte.q.values]
    allc = cols + list(Str.columns)
    MISS = ["idf_n_miss_maxw", "idf_a_miss_maxw"]
    IDF = [c for c in allc if c.startswith("idf_")]
    sets = {"v2 stage-3 (all)": [], "- idf miss_maxw (2)": MISS, "- all idf feats (10)": IDF,
            "- miss_maxw - aggregates": MISS + AGG,
            "quantised idf (0.05) - miss_maxw": MISS + ["__quant__"]}
    rng = np.random.default_rng(0)
    for country in ("US", "India"):
        ia = rng.choice(np.flatnonzero(ctr == country), 1_000_000, replace=False)
        ib = rng.choice(np.flatnonzero(cte == country), 1_000_000, replace=False)
        A = np.hstack([Xtr[ia], Str.values[ia]]); B = np.hstack([Xte[ib], Ste.values[ib]])
        X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
        perm = rng.permutation(len(y)); X, y = X[perm], y[perm]; n = int(0.8 * len(y))
        for name, drop in sets.items():
            keep = [i for i, c in enumerate(allc) if c not in drop]
            Xq = X
            if "__quant__" in drop:
                Xq = X.copy()
                qi = [i for i, c in enumerate(allc) if c.startswith("idf_")]
                Xq[:, qi] = np.round(Xq[:, qi] * 20) / 20
            Xn, Xt = Xq[:n][:, keep], Xq[n:][:, keep]
            m = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.1, verbose=-1, num_threads=8),
                          lgb.Dataset(Xn, y[:n]), 150)
            auc = roc_auc_score(y[n:], m.predict(Xt))
            imp = pd.Series(m.feature_importance("gain"), index=[allc[i] for i in keep]).sort_values(ascending=False)
            top = (imp / imp.sum()).head(6).round(3).to_dict()
            print(f"{country:6s} {name:40s} n={len(keep):3d} AUC={auc:.4f} top={top}", flush=True)


def pd_check():
    sp = json.load(open(config.work("spec_s3inv2.json")))
    spec = sp["spec"]
    X, meta, cols = exp.load("train_feat", spec["base_cols"], spec["extra"])
    m = lgb.Booster(model_file=config.work("stage2_s3inv2.txt"))
    rng = np.random.default_rng(0)
    # pairs where the decision matters: retrieval-strong candidates
    idx = rng.choice(np.flatnonzero(meta.fold.values != 0), 200_000, replace=False)
    Xs = X[idx]
    grid = {"n_tset": [60, 80, 90, 95, 98, 100], "n_ratio": [60, 80, 90, 95, 98, 100],
            "n_jw_compact": [0.7, 0.85, 0.9, 0.95, 0.98, 1.0], "a_tset": [-1, 60, 80, 90, 95, 100],
            "a_ratio": [-1, 60, 80, 90, 95, 100], "idf_n_wcov_q": [0.3, 0.6, 0.8, 0.9, 0.95, 1.0],
            "idf_a_wjacc": [0.2, 0.5, 0.7, 0.85, 0.95, 1.0], "a_first_num_eq": [-1, 0, 1],
            "n_compact_eq": [0, 1], "idf_n_miss_maxw": [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    for f, vals in grid.items():
        if f not in cols:
            continue
        j = cols.index(f)
        out = []
        for v in vals:
            Z = Xs.copy(); Z[:, j] = v
            out.append(float(m.predict(Z, num_threads=config.N_JOBS).mean()))
        mono = "increasing" if all(np.diff(out) >= -1e-3) else ("decreasing" if all(np.diff(out) <= 1e-3) else "NON-MONOTONE")
        print(f"{f:18s} " + "  ".join(f"{v}:{o:.3f}" for v, o in zip(vals, out)) + f"   -> {mono}", flush=True)


def calib():
    F = pd.read_parquet(config.work("train_feat.parquet"),
                        columns=["q", "y", "fold", "p_a_empty", "q_f_core_s1", "n_tset", "a_tset"])
    F["prob"] = np.load(config.work("oof_s3inv2.npy"))
    F = F[F.fold != 0]
    cty = pd.read_parquet(config.work("train_v1.parquet"), columns=["country"]).country.values[F.q.values]
    strong = ((F.n_tset >= 90) & (F.a_tset >= 85)).groupby(F.q).transform("sum")

    def ece(p, y, bins=15):
        b = np.minimum((p * bins).astype(int), bins - 1)
        d = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("p", "size"), p=("p", "mean"), y=("y", "mean"))
        return float((d.n * (d.p - d.y).abs()).sum() / d.n.sum())
    strata = {"all": np.ones(len(F), bool), "US": cty == "US", "India": cty == "India",
              "empty addr": F.p_a_empty.values == 1, "name shared by >=10 S1": F.q_f_core_s1.values >= 10,
              "S1 with >=5 near-copies": strong.values >= 5, "S1 with <=2 near-copies": strong.values <= 2}
    for k, m in strata.items():
        p, y = F.prob.values[m], F.y.values[m]
        mid = (p > 0.1) & (p < 0.9)
        print(f"{k:26s} n={m.sum():>11,} ECE={ece(p, y):.5f}  mid-range(0.1-0.9): n={mid.sum():,} "
              f"mean P={p[mid].mean():.3f} actual={y[mid].mean():.3f}")


if __name__ == "__main__":
    {"ids": ids, "domain": domain, "pd": pd_check, "calib": calib}[sys.argv[1]]()
