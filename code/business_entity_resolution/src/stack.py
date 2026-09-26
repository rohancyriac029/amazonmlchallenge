"""Stage 3: set-level (graph) re-scoring on top of stage-2 probabilities.

For every candidate pair (q, p) with stage-2 probability P(q, p):
  * S1 profile       : rank / max / gap / sum / #confident of P within q
  * competition      : best P of any *other* S1 for the same pool record p
  * sibling support  : similarity of p to the other confident candidates of q
                       (records of one business corroborate each other; a
                       candidate that duplicates a confident match inherits
                       evidence, one that conflicts with them loses it)
A LightGBM model over [stage-2 features + these] produces the final score.
Train-time inputs are out-of-fold stage-2 probabilities (standard stacking).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

import config
import features

_S = {}
CONF = 0.5
MAX_SIBS = 6


def _sib_block(bounds):
    s, e = bounds
    q, p, pr = _S["q"][s:e], _S["p"][s:e], _S["prob"][s:e]
    name, addr, fnum, src = _S["name"], _S["addr"], _S["fnum"], _S["src"]
    out = np.full((e - s, 9), -1.0, dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = np.arange(a, b)
        conf = idx[pr[idx] >= CONF]
        if len(conf) > MAX_SIBS:
            conf = conf[np.argsort(-pr[conf])[:MAX_SIBS]]
        for i in idx:
            pi = p[i]
            sibs = [j for j in conf if j != i]
            if not sibs:
                out[i, 8] = 0
                continue
            ns = [fuzz.token_set_ratio(name[pi], name[p[j]]) for j in sibs]
            ad = [fuzz.token_set_ratio(addr[pi], addr[p[j]]) if addr[pi] and addr[p[j]] else -1.0 for j in sibs]
            both = [min(x, y) for x, y in zip(ns, ad)]
            wboth = [pr[j] * max(z, 0) / 100 for j, z in zip(sibs, both)]
            same_num = [float(fnum[pi] != "" and fnum[pi] == fnum[p[j]]) for j in sibs]
            same_src = [float(src[pi] == src[p[j]]) for j in sibs]
            out[i] = [max(ns), float(np.mean(ns)), max(ad), float(np.mean(ad)), max(both), max(wboth),
                          max(same_num), float(np.mean(same_src)), len(sibs)]
    return out


SIB_FEATS = ["sib_name_max", "sib_name_mean", "sib_addr_max", "sib_addr_mean", "sib_both_max",
             "sib_wboth_max", "sib_same_num", "sib_same_src", "n_sibs"]


def sibling_features(df, pairs, prob, chunk_groups=20000):
    order = np.lexsort((-prob, pairs.q.values))
    q = pairs.q.values[order]
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    cuts = list(starts[::chunk_groups]) + [len(q)]
    bounds = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
    shared = {"q": q, "p": pairs.p.values[order], "prob": prob[order].astype(np.float32),
              "name": df.n_core.tolist(), "addr": df.a_norm.tolist(), "fnum": df.a_first_num.tolist(),
              "src": df.src.values}
    parts = config.parallel_map(_sib_block, bounds, "_S", shared)
    X = np.empty((len(q), len(SIB_FEATS)), dtype=np.float32)
    X[order] = np.vstack(parts)
    return pd.DataFrame(X, columns=SIB_FEATS, index=pairs.index)


def profile_features(pairs, prob):
    q, p = pairs.q.values, pairs.p.values
    prob = prob.astype(np.float32)
    X = {"prob2": prob,
         "prob_rank_q": features.group_rank(q, prob),
         "prob_max_q": features.group_max(q, prob)}
    X["prob_gap_q"] = X["prob_max_q"] - prob
    s = pd.Series(prob).groupby(q)
    X["prob_sum_q"] = s.transform("sum").values.astype(np.float32)
    X["n_conf_q"] = pd.Series(prob >= CONF).groupby(q).transform("sum").values.astype(np.float32)
    X["n_conf90_q"] = pd.Series(prob >= 0.9).groupby(q).transform("sum").values.astype(np.float32)
    # competition for the pool record: best prob of another S1
    rank_p = features.group_rank(p, prob)
    top1 = features.group_max(p, prob)
    masked = np.where(rank_p == 1, -1.0, prob).astype(np.float32)
    top2 = features.group_max(p, masked)
    X["other_best_p"] = np.where(rank_p == 1, top2, top1)
    X["prob_rank_p"] = rank_p
    X["n_conf_p"] = pd.Series(prob >= CONF).groupby(p).transform("sum").values.astype(np.float32)
    return pd.DataFrame(X, index=pairs.index)


def stack_features(df, pairs, prob):
    return pd.concat([profile_features(pairs, prob), sibling_features(df, pairs, prob)], axis=1)
