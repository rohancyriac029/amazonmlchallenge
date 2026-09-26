"""Stage-1 candidate ranker: cheap LightGBM over retrieval scores + ranks.

Prunes the ~80 retrieved candidates per S1 down to the top-N that are passed
(and reported in candidate_pairs.tsv) to the stage-2 matcher.  Features are
computed one country segment at a time (groups never cross countries because
blocking is by country), keeping peak memory bounded for 100M+ pairs.
"""
import lightgbm as lgb
import numpy as np

import config
import features

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, verbose=-1,
              num_threads=config.N_JOBS, seed=config.SEED)
DEV_FOLDS = (1, 2, 3, 4)


def _segments(seg):
    return [np.flatnonzero(seg == s) for s in np.unique(seg)]


def _X(pairs, idx):
    return features.context_features(pairs.iloc[idx]).values.astype(np.float32)


def _sample_mask(q, frac, seed):
    uq = np.unique(q)
    keep = np.random.default_rng(seed).choice(uq, max(1, int(len(uq) * frac)), replace=False)
    return np.isin(q, keep)


def _fit(X, y, rounds=200):
    return lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=rounds)


def train_models(pairs, seg, frac=0.2):
    """Returns ({fold: model} for OOF, full model)."""
    Xs, ys, fs = [], [], []
    for idx in _segments(seg):
        m = _sample_mask(pairs.q.values[idx], frac, 0)
        X = _X(pairs, idx)
        Xs.append(X[m])
        ys.append(pairs.y.values[idx][m])
        fs.append(pairs.fold.values[idx][m])
        del X
    X, y, f = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs)
    print(f"  stage-1 training sample: {len(y):,} rows, {y.sum():,} positives", flush=True)
    models = {k: _fit(X[(f != k) & (f != 0)], y[(f != k) & (f != 0)]) for k in DEV_FOLDS}
    full = _fit(X, y)
    return models, full


def oof_scores(pairs, seg, models):
    score = np.zeros(len(pairs), dtype=np.float32)
    for idx in _segments(seg):
        X = _X(pairs, idx)
        f = pairs.fold.values[idx]
        s = np.zeros(len(idx), dtype=np.float32)
        for k, m in models.items():
            te = f == k
            s[te] = m.predict(X[te], num_threads=config.N_JOBS)
            s[f == 0] += m.predict(X[f == 0], num_threads=config.N_JOBS) / len(models)
        score[idx] = s
        del X
    return score


def predict(model, pairs, seg):
    score = np.zeros(len(pairs), dtype=np.float32)
    for idx in _segments(seg):
        score[idx] = model.predict(_X(pairs, idx), num_threads=config.N_JOBS)
    return score


def top_n(pairs, score, n):
    r = features.group_rank(pairs.q.values, score)
    keep = r <= n
    out = pairs[keep].copy()
    out["s1_score"] = score[keep]
    out["s1_rank"] = r[keep]
    return out.reset_index(drop=True)
