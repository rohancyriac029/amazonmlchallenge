"""Shift-robust pipeline (v2): composition-invariant stage 2 + stage-3 stacking.

Stage-2 features exclude dataset-composition dependent inputs (raw ambiguity
counts, raw competition gaps/ranks, stage-1 score), chosen with a train-vs-test
domain classifier; optionally adds the IDF-weighted similarities of change A.

  python final2.py stack  --s2 inv2 --tag s3inv2 [--no-comp]   stage-3 OOF on dev folds + report
  python final2.py fit    --s2 inv2 --tag s3inv2               full stage-2 and stage-3 models
  python final2.py infer  --tag s3inv2 --policy expF_gate0.6   test inference, outputs, validator
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
import exp
import stack
import train
from data_loader import read_ground_truth

COMP_PROB = ["other_best_p", "prob_rank_p", "n_conf_p"]


def s2_spec(s2tag):
    """Column spec of a stage-2 experiment, reconstructed from its experiments.jsonl record."""
    rec = None
    for line in open(config.work("experiments.jsonl")):
        r = json.loads(line)
        if r.get("tag") == s2tag:
            rec = r
    allc = [c for c in train.feature_columns("train") if c not in train.META]
    inv = rec.get("invariant", "")
    drop = (([c for c in allc if "_f_" in c] if "size" in inv else [])
            + ([c for c in allc if c.endswith("_gap_p") or c.endswith("_rank_p") or c == "n_cand_p"] if "comp" in inv else [])
            + (["s1_score", "s1_rank"] if "s1" in inv else []))
    return {"base_cols": [c for c in allc if c not in drop], "extra": rec.get("extra", ""), "params": rec.get("params", "default")}


def matrix(split_table, spec, extra_df=None):
    X, meta, cols = exp.load(split_table, spec["base_cols"], spec["extra"])
    if extra_df is not None:
        X = np.hstack([X, extra_df.values.astype(np.float32)])
        cols = cols + list(extra_df.columns)
    return X, meta, cols


def stack_cmd(a):
    t0 = time.time()
    spec = s2_spec(a.s2)
    meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    p2 = np.load(config.work(f"goof_{a.s2}__normal.npy"))
    path = config.work(f"train_stack_{a.s2}.parquet")
    if os.path.exists(path):
        S = pd.read_parquet(path)
    else:
        df = pd.read_parquet(config.work("train_v1.parquet"), columns=["n_core", "a_norm", "a_first_num", "src"])
        S = stack.stack_features(df, meta, p2)
        S.to_parquet(path, index=False)
    if a.no_comp:
        S = S.drop(columns=COMP_PROB)
    X, _, cols = matrix("train_feat", spec, S)
    print(f"stage-3 features {len(cols)} ({time.time() - t0:.0f}s)", flush=True)
    prob, _ = train.oof(X, meta, cols)
    np.save(config.work(f"oof_{a.tag}.npy"), prob)
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country", "business_name"])
    pols = {k: v for k, v in train.default_policies().items() if k in ("thr0.7", "expF_gate0.6", "expF_miss0.3")}
    res, cands, q_ids = train.report(df, meta, gt, prob, a.tag, pols)
    from evaluation import paired_bootstrap
    for k, (r, _) in res.items():
        ref = config.work("scores_stack1.npy")
        if os.path.exists(ref) and k == "expF_gate0.6":
            d, lo, hi = paired_bootstrap(np.load(ref), r["_scores"])
            print(f"{a.tag} vs stack1 ({k}): {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
        train.log_experiment({"tag": a.tag, "s2": a.s2, "no_comp": a.no_comp, "policy": k, "stage": 3,
                              **{kk: v for kk, v in r.items() if not kk.startswith("_")}})
    json.dump({"s2": a.s2, "spec": spec, "no_comp": a.no_comp},
              open(config.work(f"spec_{a.tag}.json"), "w"))


def fit_cmd(a):
    sp = json.load(open(config.work(f"spec_{a.tag}.json")))
    spec = sp["spec"]
    meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    idx = train._subsample(meta.q.values, np.arange(len(meta)), 15_000_000, 7)
    X, _, cols2 = matrix("train_feat", spec)
    m2 = exp.fit(X[idx], meta.y.values[idx], cols2, spec["params"])
    m2.save_model(config.work(f"stage2_{a.tag}.txt"))
    del X
    S = pd.read_parquet(config.work(f"train_stack_{sp['s2']}.parquet"))
    if sp["no_comp"]:
        S = S.drop(columns=COMP_PROB)
    X, _, cols3 = matrix("train_feat", spec, S)
    m3 = train.fit(X[idx], meta.y.values[idx], cols3, 600)
    m3.save_model(config.work(f"stage3_{a.tag}.txt"))
    countries = sorted(pd.read_parquet(config.work("train_v1.parquet"), columns=["country"]).country.unique())
    json.dump({**sp, "cols2": cols2, "cols3": cols3, "train_countries": countries},
              open(config.work(f"spec_{a.tag}.json"), "w"))
    print(f"fit done: stage2 {len(cols2)} feats, stage3 {len(cols3)} feats, {len(idx):,} pairs")


def infer_cmd(a):
    t0 = time.time()
    from submission import write_sets, self_check
    sp = json.load(open(config.work(f"spec_{a.tag}.json")))
    spec = sp["spec"]
    if not os.path.exists(config.work("test_cands.parquet")):
        import candidates
        candidates.run("test")
    if not os.path.exists(config.work("test_feat.parquet")):
        train.build("test")                      # stage-1 pruning + pair/context features
    if spec["extra"] and not os.path.exists(config.work("test_feat_A.parquet")):
        import features_idf
        features_idf.build("test", "test_feat")
    if spec["extra"] == "A8N" and not os.path.exists(config.work("test_feat_N.parquet")):
        import features_num
        features_num.build("test", "test_feat")
    X, meta, cols2 = matrix("test_feat", spec)
    assert cols2 == sp["cols2"]
    if a.s2_avg:   # R5: mean of the 4 dev-fold stage-2 models, matching the OOF inputs stage 3 was trained on
        p2 = np.mean([lgb.Booster(model_file=config.work(f"s2fold_{a.s2_avg}_{f}.txt")).predict(X, num_threads=config.N_JOBS)
                      for f in (1, 2, 3, 4)], axis=0)
    else:
        p2 = lgb.Booster(model_file=config.work(f"stage2_{a.tag}.txt")).predict(X, num_threads=config.N_JOBS)
    df = pd.read_parquet(config.work("test_v1.parquet"), columns=["entity_id", "src", "country", "n_core",
                                                                   "a_norm", "a_first_num"])
    if a.adapt:
        import adapt
        cty = df.country.values[meta.q.values]
        unseen = [c for c in np.unique(cty) if c not in set(sp["train_countries"])]
        if unseen:
            tr = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "y"])
            src = train._subsample(tr.q.values, np.arange(len(tr)), 12_000_000, 3)
            Xs = matrix("train_feat", spec)[0][src]
            for c in unseen:
                tgt = np.flatnonzero(cty == c)
                yp = adapt.pseudo_labels(meta.q.values[tgt], meta.p.values[tgt], p2[tgt])
                print(f"adapting stage 2 to unseen {c}: pseudo pos {np.sum(yp == 1):,} neg {np.sum(yp == 0):,}", flush=True)
                m = adapt.adapt_fit(Xs, tr.y.values[src], X[tgt], yp, cols2, 600)
                p2[tgt] = m.predict(X[tgt], num_threads=config.N_JOBS)
            del Xs
    del X
    S = stack.stack_features(df, meta, p2.astype(np.float32))
    if sp["no_comp"]:
        S = S.drop(columns=COMP_PROB)
    X, _, cols3 = matrix("test_feat", spec, S)
    assert cols3 == sp["cols3"]
    meta["prob"] = lgb.Booster(model_file=config.work(f"stage3_{a.tag}.txt")).predict(X, num_threads=config.N_JOBS)
    np.save(config.work(f"test_prob_{a.tag}.npy"), meta.prob.values)
    s1_ids = df.entity_id.values[(df.src == 1).values]
    if a.policy == "cond_num":   # R1
        meta["num_agree"] = (pd.read_parquet(config.work("test_feat_N.parquet"), columns=["num_state"]).num_state.values == 1)
        sel = decision.conditional_policy(meta)
    else:
        sel = train.default_policies()[a.policy](meta)
    preds = decision.to_sets(df, sel, s1_ids)
    cands = decision.to_sets(df, meta, s1_ids)
    out = config.OUT_DIR if a.final else os.path.join(config.OUT_DIR, a.out or a.tag)
    mp_, cp_ = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    write_sets(mp_, s1_ids, preds, "matched_entity_ids")
    write_sets(cp_, s1_ids, cands, "candidate_entity_ids")
    self_check(mp_, cp_, s1_ids, df.entity_id.values[(df.src != 1).values])
    n = pd.Series([len(preds[i]) for i in s1_ids]); c = df.set_index("entity_id").country.reindex(s1_ids).values
    print("matches/S1:", n.groupby(c).mean().round(3).to_dict(), "| empty:", (n == 0).groupby(c).mean().round(4).to_dict())
    val = os.path.join(config.ROOT, "utils", "validate_submission.py")
    r = subprocess.run([sys.executable, val, "--matching", mp_, "--candidate", cp_, "--test-dir",
                        os.path.join(config.DATA_DIR, "test")], capture_output=True, text=True)
    print(r.stdout[-400:], f"\ninference done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stack", "fit", "infer"])
    ap.add_argument("--s2", default="inv2")
    ap.add_argument("--tag", default="s3inv2")
    ap.add_argument("--no-comp", action="store_true")
    ap.add_argument("--policy", default="expF_gate0.6")
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--final", action="store_true", help="write to output/ instead of output/<tag>/")
    ap.add_argument("--s2-avg", default="", help="exp tag whose saved fold models are averaged for test-time stage 2 (R5)")
    ap.add_argument("--out", default="", help="output sub-folder name (default: tag)")
    a = ap.parse_args()
    {"stack": stack_cmd, "fit": fit_cmd, "infer": infer_cmd}[a.stage](a)
