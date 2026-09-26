"""E1 evaluation: score frozen variants on clean vs distractor-injected dev fold 4.

Models are trained on dev folds 1-3 only (same sampling seed as the fold-4 OOF
models), so fold 4 is out-of-sample for every variant.  Clean scores come from
the existing 4-fold OOF predictions; injected scores from re-applying the same
models to the rebuilt candidate/feature tables of train_sim.

  python sim_eval.py v1 v2 v3        (variants to score; results appended to experiments.jsonl)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import config
import decision
import exp
import stack
import train
from data_loader import read_ground_truth
from evaluation import evaluate_predictions
from prep import fold_of

EVAL_FOLD = 4
VARIANTS = {
    # name: (stage-2 tag used for spec/stack inputs, clean OOF file of stage 3)
    "v1": ("base", "oof_stack1.npy"),
    "v2": ("inv2", "oof_s3inv2.npy"),
    "v3": ("inv3", "oof_s3inv3.npy"),
}


def spec_of(s2):
    if s2 == "base":
        cols = [c for c in train.feature_columns("train") if c not in train.META]
        return {"base_cols": cols, "extra": "", "params": "default"}
    import final2
    return final2.s2_spec(s2)


def fit_stage(X, y, q, fold, cols, params):
    tr = np.flatnonzero((fold != EVAL_FOLD) & (fold != 0))
    tr = train._subsample(q, tr, 12_000_000, EVAL_FOLD)
    return exp.fit(X[tr], y[tr], cols, params)


def main(names):
    t0 = time.time()
    gt = read_ground_truth()
    df_sim = pd.read_parquet(config.work("train_sim_v1.parquet"),
                             columns=["entity_id", "src", "country", "n_core", "a_norm", "a_first_num"])
    s1 = df_sim[(df_sim.src == 1).values]
    ids = np.sort([x for x in s1.entity_id.values if fold_of(x) == EVAL_FOLD])
    df_tr = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country"])
    meta_tr = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    for name in names:
        s2, oof_file = VARIANTS[name]
        spec = spec_of(s2)
        # --- train stage 2 / stage 3 on folds 1-3
        X, m, cols2 = exp.load("train_feat", spec["base_cols"], spec["extra"])
        M2 = fit_stage(X, m.y.values, m.q.values, m.fold.values, cols2, spec["params"])
        del X
        S = pd.read_parquet(config.work(f"train_stack_{s2}.parquet"))
        X, _, cols3 = exp.load("train_feat", spec["base_cols"], spec["extra"])
        X = np.hstack([X, S.values.astype(np.float32)]); cols3 = cols3 + list(S.columns)
        M3 = fit_stage(X, m.y.values, m.q.values, m.fold.values, cols3, "default")
        del X, S
        print(f"[{name}] models trained ({time.time() - t0:.0f}s)", flush=True)
        # --- injected table
        X, ms, _ = exp.load("train_sim_feat", spec["base_cols"], spec["extra"])
        p2 = M2.predict(X, num_threads=config.N_JOBS).astype(np.float32)
        Ssim = stack.stack_features(df_sim, ms, p2)
        X = np.hstack([X, Ssim.values.astype(np.float32)])
        ms = ms.copy(); ms["prob"] = M3.predict(X, num_threads=config.N_JOBS)
        del X
        # --- clean fold 4 (existing OOF predictions)
        mc = meta_tr[["q", "p"]].copy(); mc["prob"] = np.load(config.work(oof_file))
        np.save(config.work(f"sim_prob_{name}.npy"), ms.prob.values)
        for setname, F, dfx in (("clean", mc, df_tr), ("injected", ms, df_sim)):
            row_in = np.zeros(len(dfx), bool)           # O(n) membership via row indices, not string isin
            row_in[np.flatnonzero(dfx.entity_id.isin(set(ids)).values)] = True
            f4 = F[row_in[F.q.values]]
            unc = ((f4.prob > 0.1) & (f4.prob < 0.9)).groupby(f4.q).sum().reindex(
                np.unique(f4.q), fill_value=0).mean()
            for pol in ("expF_gate0.6", "thr0.9"):
                pred = decision.to_sets(dfx, train.default_policies()[pol](F) if pol != "thr0.9"
                                        else decision.threshold_policy(F, 0.9), ids)
                print(f"[{name}] {setname:8s} {pol:13s} uncertain/S1={unc:.3f}", end="  ")
                r = evaluate_predictions(pred, gt, ids=ids)
                train.log_experiment({"tag": f"E1_{name}", "eval": f"fold4_{setname}", "policy": pol,
                                      "uncertain_per_s1": float(unc), "note": "E1 shift suite (injected near-copies)",
                                      **{k: v for k, v in r.items() if not k.startswith("_")}})
        print(f"[{name}] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["v1", "v2"])
