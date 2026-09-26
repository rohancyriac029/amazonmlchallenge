"""Unsupervised adaptation to an unseen country via confident pseudo-labels.

Simulation (validation):  python adapt.py sim --source US --target India --base loco_us
  * source-country pairs carry real labels (dev folds only)
  * target-country pairs are scored by a source-only model (OOF of --base run),
    confident predictions become pseudo-labels (target labels are NEVER used
    except for the final score)
  * a model trained on source labels + target pseudo-labels re-scores the target

At test time the same routine is applied to countries absent from training.
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
import features
import train


def pseudo_labels(q, p, prob, hi=0.97, lo=0.03):
    """1 = confident match (best S1 for the record), 0 = confident non-match, -1 = unused."""
    best_for_p = features.group_rank(p, prob) == 1
    y = np.full(len(prob), -1, dtype=np.int8)
    y[(prob >= hi) & best_for_p] = 1
    y[prob <= lo] = 0
    return y


def adapt_fit(X_src, y_src, X_tgt, y_pseudo, cols, rounds, max_tgt=6_000_000, seed=0):
    m = y_pseudo >= 0
    idx = np.flatnonzero(m)
    if len(idx) > max_tgt:
        idx = np.random.default_rng(seed).choice(idx, max_tgt, replace=False)
    X = np.vstack([X_src, X_tgt[idx]])
    y = np.concatenate([y_src, y_pseudo[idx]])
    return train.fit(X, y, cols, rounds)


def sim(a):
    t0 = time.time()
    from data_loader import read_ground_truth
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country", "business_name"])
    meta = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    cols = [c for c in train.feature_columns("train") if c not in train.META]
    X = train.load_matrix("train", cols)
    cty = df.country.values[meta.q.values]
    base = np.load(config.work(f"oof_{a.base}.npy"))
    src = np.flatnonzero((cty == a.source) & (meta.fold.values != 0))
    src = train._subsample(meta.q.values, src, 12_000_000, 1)
    tgt = np.flatnonzero(cty == a.target)
    prob = base.copy()
    for it in range(a.iters):
        yp = pseudo_labels(meta.q.values[tgt], meta.p.values[tgt], prob[tgt], a.hi, a.lo)
        print(f"iter {it}: pseudo pos {np.sum(yp == 1):,} neg {np.sum(yp == 0):,} "
              f"(pseudo-pos precision vs truth {meta.y.values[tgt][yp == 1].mean():.4f}, "
              f"neg purity {1 - meta.y.values[tgt][yp == 0].mean():.4f})", flush=True)
        m = adapt_fit(X[src], meta.y.values[src], X[tgt], yp, cols, a.rounds)
        prob[tgt] = m.predict(X[tgt], num_threads=config.N_JOBS)
    np.save(config.work(f"oof_{a.tag}.npy"), prob)
    pol = {k: v for k, v in train.default_policies().items() if k in ("thr0.7", "expF_miss0.3", "expF_gate0.6")}
    res, cands, q_ids = train.report(df, meta, gt, prob, a.tag, pol)
    tgt_ids = set(df.entity_id.values[(df.country == a.target).values])
    ids = [i for i in q_ids if i in tgt_ids]
    for k, (r, pred) in res.items():
        print(f"   {k} on {a.target}:", end=" ")
        from evaluation import evaluate_predictions
        evaluate_predictions(pred, gt, cands, ids=ids)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sim"])
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    ap.add_argument("--base", default="loco_us")
    ap.add_argument("--tag", default="adapt_us_in")
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=600)
    sim(ap.parse_args())
