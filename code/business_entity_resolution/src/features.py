"""Pair features for (S1, S2/S3) candidate pairs.  Country agnostic by design
(country is never a feature, so unseen countries use the same model)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein, Indel

import config

_G = {}
STR_COLS = ["n_cons", "n_core", "n_compact", "n_skel", "n_legal", "n_digits",
            "a_norm", "a_words", "a_nums", "a_first_num", "a_pin"]


def _jacc(a, b):
    if not a and not b:
        return -1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _contain(a, b):
    if not a or not b:
        return -1.0
    return len(a & b) / min(len(a), len(b))


def _pair_block(bounds):
    s, e = bounds
    q = _G["q"][s:e]
    p = _G["p"][s:e]
    C = _G["cols"]
    rows = []
    for qi, pi in zip(q, p):
        a = {c: C[c][qi] for c in STR_COLS}
        b = {c: C[c][pi] for c in STR_COLS}
        ac, bc = a["n_compact"], b["n_compact"]
        at, bt = set(a["n_core"].split()), set(b["n_core"].split())
        ak, bk = set(a["n_skel"].split()), set(b["n_skel"].split())
        acons, bcons = set(a["n_cons"].split()), set(b["n_cons"].split())
        r = [
            fuzz.ratio(a["n_core"], b["n_core"]),
            fuzz.token_sort_ratio(a["n_core"], b["n_core"]),
            fuzz.token_set_ratio(a["n_core"], b["n_core"]),
            fuzz.partial_ratio(ac, bc) if ac and bc else -1,
            JaroWinkler.normalized_similarity(ac, bc),
            Levenshtein.normalized_similarity(ac, bc),
            fuzz.ratio(a["n_skel"].replace(" ", ""), b["n_skel"].replace(" ", "")),
            fuzz.ratio(a["n_cons"], b["n_cons"]),
            _jacc(at, bt),
            _contain(at, bt),
            _jacc(ak, bk),
            _jacc(acons, bcons),
            float(ac == bc and ac != ""),
            float(bool(ac) and bool(bc) and (ac in bc or bc in ac)),
            float(a["n_core"].split()[:1] == b["n_core"].split()[:1]),
            len(at), len(bt),
            min(len(ac), len(bc)) / max(len(ac), len(bc), 1),
            _jacc(set(a["n_legal"].split()), set(b["n_legal"].split())),
            float(a["n_digits"] != "" and a["n_digits"] == b["n_digits"]),
        ]
        # address
        aa, ba = a["a_norm"], b["a_norm"]
        if aa and ba:
            an, bn = set(a["a_nums"].split()), set(b["a_nums"].split())
            aw, bw = set(a["a_words"].split()), set(b["a_words"].split())
            r += [
                fuzz.ratio(aa, ba),
                fuzz.token_set_ratio(aa, ba),
                fuzz.token_sort_ratio(aa, ba),
                fuzz.partial_ratio(aa, ba),
                _jacc(aw, bw), _contain(aw, bw),
                _jacc(an, bn), _contain(an, bn),
                float(a["a_first_num"] != "" and a["a_first_num"] == b["a_first_num"]),
                float(a["a_first_num"] != "" and a["a_first_num"] in bn),
                float(b["a_first_num"] != "" and b["a_first_num"] in an),
                -1.0 if not (a["a_pin"] and b["a_pin"]) else float(bool(set(a["a_pin"].split()) & set(b["a_pin"].split()))),
                max((Indel.normalized_similarity(x, y) for x in an for y in bn if len(x) >= 2 and len(y) >= 2), default=-1.0),
                len(aw), len(bw), len(an), len(bn),
            ]
        else:
            r += [-1.0] * 17
        # name of one side appearing inside the other's address (DBA / location names)
        r.append(float(bool(bc) and len(bc) >= 5 and bc in ba.replace(" ", "")))
        rows.append(r)
    return np.asarray(rows, dtype=np.float32)


PAIR_FEATS = [
    "n_ratio", "n_tsort", "n_tset", "n_partial_compact", "n_jw_compact", "n_lev_compact",
    "n_skel_ratio", "n_cons_ratio", "n_jacc", "n_contain", "n_skel_jacc", "n_cons_jacc",
    "n_compact_eq", "n_compact_in", "n_first_tok_eq", "n_ntok_q", "n_ntok_p", "n_len_ratio",
    "n_legal_jacc", "n_digits_eq",
    "a_ratio", "a_tset", "a_tsort", "a_partial", "a_wjacc", "a_wcontain", "a_njacc", "a_ncontain",
    "a_first_num_eq", "a_qnum_in_p", "a_pnum_in_q", "a_pin_eq", "a_num_best_sim",
    "a_nw_q", "a_nw_p", "a_nn_q", "a_nn_p", "n_in_addr",
]
FREQ_COLS = ["f_core_s1", "f_core_pool", "f_addr_pool", "f_core_num_s1"]


def pair_features(df, pairs, chunk=20000):
    """df: normalized records (row index = position); pairs: DataFrame with q, p."""
    shared = {"cols": {c: df[c].tolist() for c in STR_COLS}, "q": pairs.q.values, "p": pairs.p.values}
    bounds = [(s, min(s + chunk, len(pairs))) for s in range(0, len(pairs), chunk)]
    parts = config.parallel_map(_pair_block, bounds, "_G", shared)
    X = pd.DataFrame(np.vstack(parts), columns=PAIR_FEATS, index=pairs.index)
    # record-level flags
    for c in ("n_is_domain", "n_has_dba", "n_has_phone", "n_indic", "a_empty", "a_ncomp"):
        X["p_" + c] = df[c].values[pairs.p.values]
    X["q_a_empty"] = df["a_empty"].values[pairs.q.values]
    X["q_a_ncomp"] = df["a_ncomp"].values[pairs.q.values]
    X["src3"] = (df["src"].values[pairs.p.values] == 3).astype(np.int8)
    for c in FREQ_COLS:
        if c in df:
            X["q_" + c] = df[c].values[pairs.q.values]
            X["p_" + c] = df[c].values[pairs.p.values]
    return X


def add_frequency_columns(df):
    """Unsupervised ambiguity statistics: how many S1 / pool records share a key."""
    is1 = (df.src == 1).values
    key_core = df.country.astype(str) + "|" + df.n_core
    key_addr = df.country.astype(str) + "|" + df.a_norm
    key_cn = key_core + "|" + df.a_first_num
    for name, key, mask in (("f_core_s1", key_core, is1), ("f_core_pool", key_core, ~is1),
                            ("f_addr_pool", key_addr, ~is1), ("f_core_num_s1", key_cn, is1)):
        cnt = key[mask].value_counts()
        df[name] = key.map(cnt).fillna(0).astype(np.float32).values
    df.loc[df.a_norm == "", "f_addr_pool"] = -1
    return df


# ---------------------------------------------------------------------------
# group (context) features, numpy based and float32 for 100M+ row inputs
# ---------------------------------------------------------------------------
def _starts(gs):
    return np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]


def group_rank(g, v):
    """Descending rank (1 = best, ties broken by order) of v within groups g."""
    order = np.lexsort((-v, g))
    start = _starts(g[order])
    pos = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    r = np.empty(len(g), dtype=np.float32)
    r[order] = pos + 1
    return r


def group_max(g, v):
    order = np.argsort(g, kind="stable")
    start = _starts(g[order])
    mx = np.maximum.reduceat(v[order], start)
    out = np.empty(len(g), dtype=np.float32)
    out[order] = np.repeat(mx, np.diff(np.r_[start, len(g)]))
    return out


def group_size(g):
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    return cnt[inv].astype(np.float32)


def context_features(pairs):
    """Ranks and competition within the S1 query and within the pool record."""
    q = pairs.q.values
    p = pairs.p.values
    cols = {}
    for s in [c for c in pairs.columns if c.startswith("s_")]:
        v = pairs[s].values.astype(np.float32)
        cols[s] = v
        cols[s + "_rank_q"] = group_rank(q, v)
        mq = group_max(q, v)
        cols[s + "_max_q"] = mq
        cols[s + "_gap_q"] = mq - v
        cols[s + "_rank_p"] = group_rank(p, v)
        cols[s + "_gap_p"] = group_max(p, v) - v
    cols["n_cand_q"] = group_size(q)
    cols["n_cand_p"] = group_size(p)
    return pd.DataFrame(cols, index=pairs.index)
