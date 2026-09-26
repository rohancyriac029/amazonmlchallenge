"""Generalisation experiments for stage 2 (never touches holdout labels).

Each run trains out-of-fold on dev folds (1-4) and reports macro F0.5 on
  * normal dev  : folds 1-4 at training density
  * dense dev   : folds 1-4 with S1-dropout (test-like distractor density)
  * LOCO        : optional --loco SRC:TGT, train on SRC only, score TGT only
Per-entity scores are saved for paired bootstraps (--compare) and every result
is appended to work/experiments.jsonl.

  python exp.py --tag g_base --train normal
  python exp.py --tag g_dense --train dense --compare g_base
  python exp.py --tag g_denseA --train dense --extra A --compare g_dense
  python exp.py --tag g_denseAB --train dense --extra A --params reg --compare g_denseA
  python exp.py --tag g_base_loco_us --train normal --loco US:India
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import decision
import train
from data_loader import read_ground_truth
from evaluation import evaluate_predictions, paired_bootstrap

TABLES = {"normal": "train_feat", "dense": "train_dense_feat", "univ": "train_univ_feat"}
# features whose scale depends on the size / crowding of the dataset (domain-classifier diagnosis)
SIZE_RAW = ["q_f_core_s1", "p_f_core_s1", "q_f_core_pool", "p_f_core_pool", "q_f_core_num_s1",
            "p_f_core_num_s1", "q_f_addr_pool", "p_f_addr_pool"]
COMPETITION_RAW = ["n_cand_p", "s_name_gap_p", "s_addr_gap_p", "s_combo_gap_p", "s_conj_gap_p", "s_gram_gap_p"]
REG = dict(num_leaves=63, min_data_in_leaf=1000, lambda_l2=10.0, feature_fraction=0.6, bagging_fraction=0.7)
MONO_UP = ["n_ratio", "n_tsort", "n_tset", "n_jw_compact", "n_lev_compact", "n_skel_ratio", "n_cons_ratio",
           "n_jacc", "n_contain", "n_skel_jacc", "n_cons_jacc", "n_compact_eq", "s_name", "s_combo", "s_conj",
           "s_gram", "s1_score", "idf_n_wjacc", "idf_n_wcov_q", "idf_n_wcov_p", "idf_n_soft"]
MONO_DOWN = ["s1_rank"]
POLICIES = ("expF_gate0.6", "thr0.7")


def load(table, cols, extra):
    import pyarrow.parquet as pq
    path = config.work(f"{table}.parquet")
    names = pq.read_schema(path).names
    meta = pd.read_parquet(path, columns=[c for c in ("q", "p", "y", "fold") if c in names])
    n = len(meta)
    ext_cols = []
    num_cols = []
    if extra in ("A", "A10", "A8", "A8N"):
        import features_idf
        ext_cols = {"A": features_idf.A_FEATS, "A10": features_idf.A_FEATS[:10],
                    "A8": [c for c in features_idf.A_FEATS[:10] if not c.endswith("miss_maxw")],
                    "A8N": [c for c in features_idf.A_FEATS[:10] if not c.endswith("miss_maxw")]}[extra]
    if extra == "A8N":
        import features_num
        num_cols = features_num.N_FEATS
    X = np.empty((n, len(cols) + len(ext_cols) + len(num_cols)), dtype=np.float32)
    for s in range(0, len(cols), 12):
        cb = cols[s:s + 12]
        t = pq.read_table(path, columns=cb)
        for j, c in enumerate(cb):
            X[:, s + j] = t.column(c).to_numpy()
    if ext_cols:
        t = pq.read_table(config.work(f"{table}_A.parquet"), columns=ext_cols)
        for j, c in enumerate(ext_cols):
            X[:, len(cols) + j] = t.column(c).to_numpy()
    if num_cols:
        t = pq.read_table(config.work(f"{table}_N.parquet"), columns=num_cols)
        for j, c in enumerate(num_cols):
            X[:, len(cols) + len(ext_cols) + j] = t.column(c).to_numpy()
    return X, meta, cols + ["idf_" + c for c in ext_cols] + num_cols


def fit(X, y, cols, params, w=None):
    p = dict(train.PARAMS)
    if params == "reg":
        p.update(REG)
        p["monotone_constraints"] = [1 if c in MONO_UP else (-1 if c in MONO_DOWN else 0) for c in cols]
        p["monotone_constraints_method"] = "intermediate"
    return lgb.train(p, lgb.Dataset(X, y, weight=w, feature_name=cols), num_boost_round=600)


def sample_rows(meta, mask, n, seed):
    idx = np.flatnonzero(mask)
    return train._subsample(meta.q.values, idx, n, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", default="normal", help="comma list of tables: normal,dense,univ")
    ap.add_argument("--evals", default="normal,univ", help="comma list of tables to evaluate on")
    ap.add_argument("--invariant", default="", choices=["", "size", "size+comp", "size+comp+s1"],
                    help="drop dataset-size dependent raw features (size) and raw competition gaps (comp)")
    ap.add_argument("--extra", default="", choices=["", "A", "A10", "A8", "A8N"],
                    help="A: IDF similarities + relative counts; A10: IDF similarities only; "
                         "A8: A10 without the two max-unmatched-token weights (scale-shifted, see audit)")
    ap.add_argument("--params", default="default", choices=["default", "reg"])
    ap.add_argument("--loco", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--balance", action="store_true", help="country-balanced sample weights")
    a = ap.parse_args()
    t0 = time.time()
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country"])
    base_cols = [c for c in train.feature_columns("train") if c not in train.META]
    allc = [c for c in train.feature_columns("train") if c not in train.META]
    comp_all = [c for c in allc if c.endswith("_gap_p") or c.endswith("_rank_p") or c == "n_cand_p"]
    size_all = [c for c in allc if "_f_" in c]
    drop = ((size_all if "size" in a.invariant else []) + (comp_all if "comp" in a.invariant else [])
            + (["s1_score", "s1_rank"] if "s1" in a.invariant else []))
    base_cols = [c for c in base_cols if c not in drop]
    train_sets = a.train.split(",")
    evals = a.evals.split(",")
    data = {k: load(TABLES[k], base_cols, a.extra) for k in dict.fromkeys(train_sets + evals)}
    cols = next(iter(data.values()))[2]
    print(f"{len(cols)} features; loaded ({time.time() - t0:.0f}s)", flush=True)
    src, tgt = (a.loco.split(":") + [None])[:2] if a.loco else (None, None)
    cty = {k: df.country.values[m.q.values] for k, (_, m, _) in data.items()}
    per_set = 12_000_000 // len(train_sets)
    prob = {k: np.zeros(len(m), dtype=np.float32) for k, (_, m, _) in data.items()}

    def train_on(fold_mask_fn, seed):
        Xs, ys, ws = [], [], []
        for k in train_sets:
            X, m, _ = data[k]
            mask = fold_mask_fn(m) & ((cty[k] == src) if src else True)
            idx = sample_rows(m, mask, per_set, seed)
            Xs.append(X[idx]); ys.append(m.y.values[idx])
            if a.balance:
                c = cty[k][idx]
                share = pd.Series(c).map(pd.Series(c).value_counts(normalize=True)).values
                ws.append((1.0 / share).astype(np.float32))
        w = np.concatenate(ws) if ws else None
        return fit(np.vstack(Xs), np.concatenate(ys), cols, a.params, w)

    if src:
        mdl = train_on(lambda m: m.fold.values != 0, 1)
        for k, (X, m, _) in data.items():
            te = np.flatnonzero((cty[k] == tgt) & (m.fold.values != 0))
            prob[k][te] = mdl.predict(X[te], num_threads=config.N_JOBS)
    else:
        for f in train.DEV_FOLDS:
            mdl = train_on(lambda m, f=f: (m.fold.values != f) & (m.fold.values != 0), f)
            for k, (X, m, _) in data.items():
                te = np.flatnonzero(m.fold.values == f)
                prob[k][te] = mdl.predict(X[te], num_threads=config.N_JOBS)
                ho = np.flatnonzero(m.fold.values == 0)     # competition context only, never scored
                prob[k][ho] += mdl.predict(X[ho], num_threads=config.N_JOBS) / len(train.DEV_FOLDS)
            print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    for k in prob:
        np.save(config.work(f"goof_{a.tag}__{k}.npy"), prob[k])
    imp = pd.Series(mdl.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:", (imp / imp.sum()).head(12).round(3).to_dict())
    # evaluation
    from prep import fold_of
    s1 = df[df.src == 1]
    s1_fold = np.array([fold_of(x) for x in s1.entity_id.values])
    import dense as D
    dropped = set(df.entity_id.values[D.dropped_s1(df)])
    univ_gone = set(df.entity_id.values[D.universe_removed(df)]) if "univ" in evals else set()
    for k in evals:
        X, m, _ = data[k]
        ids = s1.entity_id.values[(s1_fold != 0) & ((s1.country.values == tgt) if tgt else True)]
        if k == "dense":
            ids = np.array([i for i in ids if i not in dropped])
        if k == "univ":
            ids = np.array([i for i in ids if i not in univ_gone])
        ids = np.sort(ids)
        F = m[["q", "p"]].copy(); F["prob"] = prob[k]
        cands = decision.to_sets(df, F, ids)
        for pol in POLICIES:
            pred = decision.to_sets(df, train.default_policies()[pol](F), ids)
            print(f"[{a.tag}] eval={k} policy={pol}:", end=" ")
            r = evaluate_predictions(pred, gt, cands, ids=ids, return_scores=True)
            key = f"{a.tag}__{k}__{pol}"
            np.save(config.work(f"gscores_{key}.npy"), r["_scores"])
            rec = {"tag": a.tag, "eval": k, "policy": pol, "train": a.train, "extra": a.extra, "params": a.params,
                   "invariant": a.invariant,
                   "loco": a.loco, "balance": a.balance, "seed": config.SEED, "model": "lightgbm-stage2",
                   "blocking": "5ch hashed tfidf + stage1 top25", "n_features": len(cols),
                   **{kk: v for kk, v in r.items() if not kk.startswith("_")}}
            if a.compare:
                cp = config.work(f"gscores_{a.compare}__{k}__{pol}.npy")
                if os.path.exists(cp):
                    d, lo, hi = paired_bootstrap(np.load(cp), r["_scores"])
                    rec["vs"] = a.compare; rec["delta"] = d; rec["ci"] = [lo, hi]
                    print(f"      vs {a.compare}: {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
            train.log_experiment(rec)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
