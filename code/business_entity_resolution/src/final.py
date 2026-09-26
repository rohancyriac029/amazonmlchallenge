"""Final stages.

  python final.py holdout --tag T --policy P   one-time score on the untouched holdout fold 0
  python final.py fit --base B --tag T         stage-2 and stage-3 models on all training data
  python final.py infer --policy P             test inference -> output/*.tsv + validator

Holdout predictions in oof_<tag>.npy come exclusively from models trained on the
dev folds (the mean of the four fold models at stage 2 and at stage 3), so the
holdout is never seen by any model, threshold or policy choice.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import decision
import stack
import train
from data_loader import read_ground_truth


def holdout(a):
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country", "business_name"])
    meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    prob = np.load(config.work(f"oof_{a.tag}.npy"))
    pol = {a.policy: train.default_policies()[a.policy]}
    res, cands, q_ids = train.report(df, meta, gt, prob, "HOLDOUT", pol, eval_folds=(0,))
    r = res[a.policy][0]
    train.subset_report(df[df.entity_id.isin(set(q_ids)) | (df.src != 1)], gt, res[a.policy][1], cands)
    train.log_experiment({"tag": a.tag + "_HOLDOUT", "policy": a.policy,
                          **{k: v for k, v in r.items() if not k.startswith("_")}})


def _stage2_cols():
    return [c for c in train.feature_columns("train") if c not in train.META]


def fit(a):
    """Stage 2 on all folds; stage 3 on all folds using OOF stage-2 based stack features."""
    meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    cols2 = _stage2_cols()
    X = train.load_matrix("train", cols2)
    idx = train._subsample(meta.q.values, np.arange(len(meta)), 15_000_000, 7)
    m2 = train.fit(X[idx], meta.y.values[idx], cols2, a.rounds)
    m2.save_model(config.work("stage2_model.txt"))
    del X
    S = train.load_stack_features(meta, a.base)
    X = train.load_matrix("train", cols2, S)
    cols3 = cols2 + list(S.columns)
    del S
    m3 = train.fit(X[idx], meta.y.values[idx], cols3, a.rounds)
    m3.save_model(config.work("stage3_model.txt"))
    countries = sorted(pd.read_parquet(config.work("train_v1.parquet"), columns=["country"]).country.unique())
    with open(config.work("final_cols.json"), "w") as f:
        json.dump({"stage2": cols2, "stage3": cols3, "train_countries": countries}, f)
    print(f"stage-2/3 models trained on {len(idx):,} pairs; {len(cols2)} / {len(cols3)} features")


def adapt_unseen(df, meta, prob2, cols):
    """Pseudo-label domain adaptation of stage 2 for test countries absent from
    training (validated by the US->India simulation in adapt.py)."""
    import adapt
    cty = df.country.values[meta.q.values]
    unseen = [c for c in np.unique(cty) if c not in set(cols["train_countries"])]
    if not unseen:
        return prob2
    tr_meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "y"])
    src = train._subsample(tr_meta.q.values, np.arange(len(tr_meta)), 12_000_000, 3)
    Xs = train.load_matrix("train", cols["stage2"])[src]
    ys = tr_meta.y.values[src]
    del tr_meta
    Xt_all = train.load_matrix("test", cols["stage2"])
    prob2 = prob2.copy()
    for c in unseen:
        tgt = np.flatnonzero(cty == c)
        yp = adapt.pseudo_labels(meta.q.values[tgt], meta.p.values[tgt], prob2[tgt])
        print(f"adapting to unseen country {c}: {len(tgt):,} pairs, pseudo pos {np.sum(yp == 1):,} "
              f"neg {np.sum(yp == 0):,}", flush=True)
        m = adapt.adapt_fit(Xs, ys, Xt_all[tgt], yp, cols["stage2"], 600)
        prob2[tgt] = m.predict(Xt_all[tgt], num_threads=config.N_JOBS)
    return prob2


def infer(a):
    t0 = time.time()
    import candidates
    from submission import write_sets, self_check
    if not os.path.exists(config.work("test_cands.parquet")):
        candidates.run("test")
    if not os.path.exists(config.work("test_feat.parquet")):
        train.build("test")
    cols = json.load(open(config.work("final_cols.json")))
    meta = pd.read_parquet(config.work("test_feat.parquet"), columns=["q", "p"])
    X = train.load_matrix("test", cols["stage2"])
    prob2 = lgb.Booster(model_file=config.work("stage2_model.txt")).predict(X, num_threads=config.N_JOBS)
    del X
    print(f"stage-2 scored {len(meta):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    df = pd.read_parquet(config.work("test_v1.parquet"), columns=["entity_id", "src", "country", "n_core",
                                                                   "a_norm", "a_first_num"])
    if not a.no_adapt:
        prob2 = adapt_unseen(df, meta, prob2, cols)
    S = stack.stack_features(df, meta, prob2.astype(np.float32))
    X = train.load_matrix("test", cols["stage2"], S)
    del S
    meta["prob"] = lgb.Booster(model_file=config.work("stage3_model.txt")).predict(X, num_threads=config.N_JOBS)
    del X
    np.save(config.work("test_prob.npy"), meta.prob.values)
    print(f"stage-3 scored ({time.time() - t0:.0f}s)", flush=True)
    s1_ids = df.entity_id.values[(df.src == 1).values]
    sel = train.default_policies()[a.policy](meta)
    preds = decision.to_sets(df, sel, s1_ids)
    cands = decision.to_sets(df, meta, s1_ids)
    out = config.OUT_DIR
    mp_, cp_ = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    write_sets(mp_, s1_ids, preds, "matched_entity_ids")
    write_sets(cp_, s1_ids, cands, "candidate_entity_ids")
    self_check(mp_, cp_, s1_ids, df.entity_id.values[(df.src != 1).values])
    cty = df.set_index("entity_id").country.reindex(s1_ids).values
    n = pd.Series([len(preds[i]) for i in s1_ids])
    print("predicted matches per S1 by country:", n.groupby(cty).mean().round(3).to_dict())
    print("predicted-empty rate by country:", (n == 0).groupby(cty).mean().round(4).to_dict())
    val = os.path.join(config.ROOT, "utils", "validate_submission.py")
    if os.path.exists(val):
        r = subprocess.run([sys.executable, val, "--matching", mp_, "--candidate", cp_,
                            "--test-dir", os.path.join(config.DATA_DIR, "test")], capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-2000:])
    print(f"inference done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["holdout", "fit", "infer"])
    ap.add_argument("--tag", default="stack1")
    ap.add_argument("--base", default="base", help="stage-2 OOF tag used for stack features")
    ap.add_argument("--policy", default="expF_gate0.6")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--no-adapt", action="store_true", help="disable unseen-country adaptation")
    a = ap.parse_args()
    {"holdout": holdout, "fit": fit, "infer": infer}[a.stage](a)
