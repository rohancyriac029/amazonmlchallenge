"""Step 3: features, 5-fold out-of-fold LightGBM, decision-policy evaluation, final model."""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import decision
import features
from data_loader import read_ground_truth
from evaluation import evaluate_predictions


def fused_rank(pairs):
    """Best (minimum) within-S1 rank over all retrieval channels; ties by s_combo."""
    sc = [c for c in pairs.columns if c.startswith("s_")]
    best = np.full(len(pairs), np.inf)
    for c in sc:
        best = np.minimum(best, pairs.groupby("q")[c].rank(ascending=False, method="first").values)
    tmp = pd.DataFrame({"q": pairs.q.values, "k": best - 1e-3 * pairs.s_combo.values})
    return tmp.groupby("q").k.rank(method="first").values


def prune(pairs, n_keep):
    r = fused_rank(pairs)
    pairs = pairs.assign(fused_rank=r.astype(np.float32))
    return pairs[r <= n_keep].reset_index(drop=True)


def build(split, n_keep=25, force=False):
    """Stage-1 prune to top-n_keep per S1, then compute stage-2 features."""
    path = config.work(f"{split}_feat.parquet")
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    import ranker1
    t0 = time.time()
    need = sorted(set(features.STR_COLS) | {"entity_id", "src", "country", "n_is_domain", "n_has_dba",
                                            "n_has_phone", "n_indic", "a_empty", "a_ncomp"})
    df = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=need)
    df = features.add_frequency_columns(df)
    pairs = pd.read_parquet(config.work(f"{split}_cands.parquet"))
    mpath = config.work("stage1_model.txt")
    seg = df.country.values[pairs.q.values]
    if split == "train":
        models, full = ranker1.train_models(pairs, seg)
        full.save_model(mpath)
        score = ranker1.oof_scores(pairs, seg, models)
    else:
        score = ranker1.predict(lgb.Booster(model_file=mpath), pairs, seg)
    n_before = len(pairs)
    if "y" in pairs:
        r = features.group_rank(pairs.q.values, score)
        for n in (10, 15, 20, 25, 30, 40):
            print(f"  stage-1 top{n}: positives kept {pairs.y.values[r <= n].sum():,} / {pairs.y.sum():,}", flush=True)
    pairs = ranker1.top_n(pairs, score, n_keep)
    print(f"pruned {n_before:,} -> {len(pairs):,} pairs ({time.time() - t0:.0f}s)")
    X = features.pair_features(df, pairs)
    C = features.context_features(pairs.drop(columns=["s1_score", "s1_rank"]))
    out = pd.concat([pairs, C.drop(columns=[c for c in C.columns if c in pairs.columns]), X], axis=1)
    print(f"features {out.shape} ({time.time() - t0:.0f}s)")
    out.to_parquet(path, index=False)
    return out


META = {"q", "p", "y", "fold"}


def feature_columns(split):
    import pyarrow.parquet as pq
    return pq.read_schema(config.work(f"{split}_feat.parquet")).names


def load_matrix(split, cols, extra=None, batch=12):
    """Fill a float32 matrix column-batch by column-batch from the feature parquet
    (avoids holding the full DataFrame and the matrix at the same time)."""
    import pyarrow.parquet as pq
    path = config.work(f"{split}_feat.parquet")
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    n_extra = 0 if extra is None else extra.shape[1]
    X = np.empty((n, len(cols) + n_extra), dtype=np.float32)
    for s in range(0, len(cols), batch):
        cb = cols[s:s + batch]
        t = pq.read_table(path, columns=cb)
        for j, c in enumerate(cb):
            X[:, s + j] = t.column(c).to_numpy()
        del t
    if extra is not None:
        X[:, len(cols):] = extra.values
    return X

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=config.N_JOBS, seed=config.SEED)


def feat_cols(F, drop=()):
    return [c for c in F.columns if c not in META and c not in drop]


def fit(X, y, cols, rounds=600, weight=None):
    ds = lgb.Dataset(X, y, weight=weight, feature_name=cols, free_raw_data=True)
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


DEV_FOLDS = (1, 2, 3, 4)   # fold 0 = untouched final holdout (see prep.HOLDOUT_FOLD)


def _subsample(q, idx, max_rows, seed):
    if len(idx) <= max_rows:
        return idx
    qs = np.unique(q[idx])
    seed = seed if config.SEED == 42 else seed + 1000 * config.SEED   # default keeps earlier runs reproducible
    keep_q = np.random.default_rng(seed).choice(qs, int(len(qs) * max_rows / len(idx)), replace=False)
    return idx[np.isin(q[idx], keep_q)]


def oof(X, meta, cols, max_train_rows=12_000_000, rounds=600, weight=None, train_mask=None):
    """4-fold OOF over dev folds.  Holdout-fold pairs get the mean of the fold
    models (never trained on them; used only as competition context)."""
    fold = meta.fold.values
    y = meta.y.values
    q = meta.q.values
    prob = np.zeros(len(meta), dtype=np.float32)
    ho = np.flatnonzero(fold == 0)
    for k in DEV_FOLDS:
        tr_m = (fold != k) & (fold != 0)
        if train_mask is not None:
            tr_m &= train_mask
        tr = np.flatnonzero(tr_m)
        tr = _subsample(q, tr, max_train_rows, k)
        m = fit(X[tr], y[tr], cols, rounds, None if weight is None else weight[tr])
        te = np.flatnonzero(fold == k)
        prob[te] = m.predict(X[te], num_threads=config.N_JOBS)
        prob[ho] += m.predict(X[ho], num_threads=config.N_JOBS) / len(DEV_FOLDS)
        print(f"  fold {k}: train {len(tr):,} test {len(te):,}", flush=True)
    return prob, m


def report(df, F, gt, prob, tag, policies, eval_folds=DEV_FOLDS):
    F = F[["q", "p", "y", "fold"]].copy()
    F["prob"] = prob
    s1 = df[df.src == 1]
    from prep import fold_of
    s1_fold = np.array([fold_of(x) for x in s1.entity_id.values])
    all_ids = s1.entity_id.values
    q_ids = all_ids[np.isin(s1_fold, eval_folds)]
    fold_ids = {k: all_ids[s1_fold == k] for k in eval_folds}
    cands = decision.to_sets(df, F, all_ids)
    results = {}
    for name, fn in policies.items():
        sel = fn(F)
        pred = decision.to_sets(df, sel, all_ids)
        print(f"[{tag}] policy={name}")
        r = evaluate_predictions(pred, gt, cands, ids=q_ids, return_scores=True)
        per_fold = [evaluate_predictions(pred, gt, None, ids=fold_ids[k], verbose=False)["macro_f05"]
                    for k in eval_folds]
        r["fold_mean"], r["fold_std"] = float(np.mean(per_fold)), float(np.std(per_fold))
        print(f"      per-fold F0.5 mean={r['fold_mean']:.5f} std={r['fold_std']:.5f}")
        results[name] = (r, pred)
    return results, cands, q_ids


def load_stack_features(meta, base_tag):
    import stack
    path = config.work(f"train_stack_{base_tag}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["n_core", "a_norm", "a_first_num", "src"])
    S = stack.stack_features(df, meta, np.load(config.work(f"oof_{base_tag}.npy")))
    S.to_parquet(path, index=False)
    print(f"stack features {S.shape} ({time.time() - t0:.0f}s)", flush=True)
    return S


def log_experiment(rec):
    with open(config.work("experiments.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def subset_report(df, gt, pred, cands):
    s1 = df[df.src == 1]
    by_c = s1.groupby("country").entity_id.apply(list)
    for c, ids in by_c.items():
        print(f"   country={c}:", end=" ")
        evaluate_predictions(pred, gt, cands, ids=ids)
    for k in (2, 3):
        pk = {a: {x for x in v if x.startswith(f"S{k}")} for a, v in pred.items()}
        gk = {a: {x for x in v if x.startswith(f"S{k}")} for a, v in gt.items()}
        ck = {a: {x for x in v if x.startswith(f"S{k}")} for a, v in cands.items()}
        print(f"   only S{k}:", end=" ")
        evaluate_predictions(pk, gk, ck, ids=list(s1.entity_id))
    single = [i for i in s1.entity_id if not gt[i]]
    multi = [i for i in s1.entity_id if len(gt[i]) >= 2]
    print("   singletons:", end=" ")
    evaluate_predictions(pred, gt, cands, ids=single)
    print("   multi-match:", end=" ")
    evaluate_predictions(pred, gt, cands, ids=multi)
    # ambiguous names (name shared by several S1)
    nm = s1.business_name.str.lower()
    amb = s1.entity_id[nm.map(nm.value_counts()) > 1].tolist()
    print("   ambiguous-name S1:", end=" ")
    evaluate_predictions(pred, gt, cands, ids=amb)


def default_policies():
    pol = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pol[f"thr{t}"] = (lambda F, t=t: decision.threshold_policy(F, t))
    pol["thr0.5_noexcl"] = lambda F: decision.threshold_policy(F, 0.5, excl=False)
    for mm in (0.0, 0.3):
        pol[f"expF_miss{mm}"] = (lambda F, mm=mm: decision.expected_f_policy(F, miss_mass=mm))
    pol["expF_gate0.6"] = lambda F: decision.gated_expected_f_policy(F, gate=0.6, miss_mass=0.3)
    pol["thr0.9"] = lambda F: decision.threshold_policy(F, 0.9)   # final: robust to near-copy distractor density
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--drop", default="")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--compare", default="", help="tag of a previous experiment for paired bootstrap")
    ap.add_argument("--note", default="")
    ap.add_argument("--train-country", default="", help="leave-one-country-out: train only on this country")
    ap.add_argument("--stack", default="", help="tag of a stage-2 OOF run; adds stage-3 set-level features")
    a = ap.parse_args()
    t0 = time.time()
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country", "business_name"])
    if a.rebuild or not os.path.exists(config.work("train_feat.parquet")):
        build("train", force=True)
    allc = feature_columns("train")
    cols = [c for c in allc if c not in META and c not in [d for d in a.drop.split(",") if d]]
    meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    print(f"{len(cols)} features; pairs {len(meta):,}; pos {meta.y.sum():,} ({time.time() - t0:.0f}s)")
    S = load_stack_features(meta, a.stack) if a.stack else None
    X = load_matrix("train", cols, S)
    if S is not None:
        cols = cols + list(S.columns)
        del S
    import gc
    gc.collect()
    train_mask = None
    if a.train_country:
        train_mask = (df.country.values[meta.q.values] == a.train_country)
        print(f"LOCO: training only on {a.train_country} ({train_mask.sum():,} pairs)")
    prob, model = oof(X, meta, cols, rounds=a.rounds, train_mask=train_mask)
    F = meta
    np.save(config.work(f"oof_{a.tag}.npy"), prob)
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:\n", (imp / imp.sum()).head(25).round(4).to_string())
    res, cands, q_ids = report(df, F, gt, prob, a.tag, default_policies())
    best = max(res, key=lambda k: res[k][0]["macro_f05"])
    r = res[best][0]
    print(f"BEST policy {best}: {r['macro_f05']:.5f}")
    np.save(config.work(f"scores_{a.tag}.npy"), r["_scores"])
    if a.compare and os.path.exists(config.work(f"scores_{a.compare}.npy")):
        from evaluation import paired_bootstrap
        d, lo, hi = paired_bootstrap(np.load(config.work(f"scores_{a.compare}.npy")), r["_scores"])
        print(f"paired bootstrap {a.tag} - {a.compare}: {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    subset_report(df[df.entity_id.isin(set(q_ids)) | (df.src != 1)], gt, res[best][1], cands)
    log_experiment({"tag": a.tag, "note": a.note, "stack": a.stack, "train_country": a.train_country, "n_features": len(cols), "dropped": a.drop,
                    "rounds": a.rounds, "policy": best, "seed": config.SEED, "model": "lightgbm",
                    "pairs": int(len(F)),
                    **{k: v for k, v in r.items() if not k.startswith("_")},
                    "all_policies": {k: v[0]["macro_f05"] for k, v in res.items()}})
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
