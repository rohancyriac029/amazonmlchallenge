"""Candidate generation: multi-channel pruned sparse TF-IDF retrieval, blocked by country.

Channels (each returns top-K pool records per S1 query):
  name  : core tokens, phonetic skeleton tokens, adjacent token bigrams,
          compact-name prefixes / full compact string (typo + spacing + domain robust)
  addr  : address word tokens, numbers, number+word and word bigrams
  combo : name and address features in one vector (joint evidence)
Features whose document frequency in the pool exceeds `max_df` are dropped
(they carry little identity information and dominate cost); bigrams restore
specificity for combinations of common words.
"""
import gc
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config


def _name_feats(core, skel, compact, cons):
    f = []
    toks = core.split()
    f += ["w:" + t for t in toks]
    f += ["k:" + t for t in skel.split() if len(t) >= 2]
    f += ["b:" + a + "_" + b for a, b in zip(toks, toks[1:])]
    if len(compact) >= 4:
        f.append("p:" + compact[:5])
        f.append("s:" + compact[-5:])
        f.append("c:" + compact)
    # conservative form catches names whose core became empty / legal only
    if not toks:
        f += ["w:" + t for t in cons.split()]
    return f


def _addr_feats(words, nums, norm):
    f = []
    wt = words.split()
    f += ["a:" + t for t in wt]
    f += ["n:" + t for t in nums.split() if len(t) >= 2]
    f += ["ab:" + a + "_" + b for a, b in zip(wt, wt[1:])]
    nt = norm.split()
    for a, b in zip(nt, nt[1:]):
        if a.isdigit() and not b.isdigit():
            f.append("nw:" + a + "_" + b)
    return f


def _conj_feats(core, words, nums):
    """Name-token x locality conjunctions: identity = name + place."""
    nt = [t for t in core.split() if len(t) >= 2][:4]
    aw = [t for t in words.split() if len(t) >= 3][:12]
    an = [t for t in nums.split()][:4]
    f = ["x:" + a + "|" + b for a in nt for b in aw]
    f += ["y:" + a + "|" + b for a in nt for b in an]
    return f


def _gram_feats(compact, n=4):
    c = "^" + compact + "$"
    return list({"g:" + c[i:i + n] for i in range(max(1, len(c) - n + 1))})


def build_feature_lists(df, channel):
    if channel == "conj":
        return [_conj_feats(a, b, c) for a, b, c in zip(df.n_core, df.a_words, df.a_nums)]
    if channel == "gram":
        return [_gram_feats(a) for a in df.n_compact]
    if channel == "name":
        return [_name_feats(a, b, c, d) for a, b, c, d in zip(df.n_core, df.n_skel, df.n_compact, df.n_cons)]
    if channel == "addr":
        return [_addr_feats(a, b, c) for a, b, c in zip(df.a_words, df.a_nums, df.a_norm)]
    raise ValueError(channel)


N_BUCKETS = 1 << 26
_BG = {}


def _hash_chunk(bounds):
    s, e = bounds
    sub = _BG["df"].iloc[s:e]
    lists = build_feature_lists(sub, _BG["channel"])
    counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    flat = np.array([f for fl in lists for f in fl], dtype=object)
    if len(flat) == 0:
        return counts, np.zeros(0, dtype=np.int32)
    h = (pd.util.hash_array(flat, categorize=False) & np.uint64(N_BUCKETS - 1)).astype(np.int32)
    return counts, h


def hashed_matrix(df, channel, chunk=100000):
    """Binary CSR (n x N_BUCKETS) of hashed features, built in parallel."""
    bounds = [(s, min(s + chunk, len(df))) for s in range(0, len(df), chunk)]
    parts = config.parallel_map(_hash_chunk, bounds, "_BG", {"df": df, "channel": channel})
    counts = np.concatenate([c for c, _ in parts])
    idx = np.concatenate([h for _, h in parts])
    indptr = np.r_[0, np.cumsum(counts)]
    M = sp.csr_matrix((np.ones(len(idx), dtype=np.float32), idx, indptr), shape=(len(df), N_BUCKETS))
    M.sum_duplicates()
    M.data[:] = 1.0
    return M


def tfidf_hashed(Q, P, max_df):
    """IDF from query+pool document frequencies; drop df<2 or df>max_df; L2 normalise."""
    dfreq = np.bincount(Q.indices, minlength=N_BUCKETS) + np.bincount(P.indices, minlength=N_BUCKETS)
    n = Q.shape[0] + P.shape[0]
    idf = np.where((dfreq >= 2) & (dfreq <= max_df), np.log(n / np.maximum(dfreq, 1)), 0).astype(np.float32)
    out = []
    for M in (Q, P):
        M = M.copy()
        M.data = idf[M.indices]
        M.eliminate_zeros()
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        norms[norms == 0] = 1
        M.data /= np.repeat(norms, np.diff(M.indptr)).astype(np.float32)
        out.append(M)
    return out


def topk(Q, P, k, n_threads=None, batch=200000):
    n_threads = n_threads or config.N_JOBS
    PT = P.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, Q.shape[0], batch):
        C = sp_matmul_topn(Q[s:s + batch], PT, top_n=k, threshold=1e-6, n_threads=n_threads).tocoo()
        rows.append(C.row + s)
        cols.append(C.col)
        vals.append(C.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


DEFAULT_CHANNELS = {
    # name: (feature channels combined with weights, top-k)
    "name": ({"name": 1.0}, 30),
    "addr": ({"addr": 1.0}, 30),
    "combo": ({"name": 0.5, "addr": 0.5}, 30),
    "conj": ({"conj": 0.6, "name": 0.2, "addr": 0.2}, 30),
    "gram": ({"gram": 1.0}, 20),
}


def _combine(mats, weights, side):
    blocks = [mats[f][side] * np.float32(np.sqrt(w)) for f, w in weights.items()]
    return blocks[0].tocsr() if len(blocks) == 1 else sp.hstack(blocks).tocsr()


def generate_candidates(df, q_mask, channels=None, max_df=10000, verbose=True):
    """Return DataFrame(q, p, s_<channel>...) with row indices into df.

    q_mask: boolean mask selecting S1 query rows. Pool = all S2/S3 rows.
    Blocking by exact country string (open set).
    """
    channels = channels or DEFAULT_CHANNELS
    out = []
    df = df.reset_index(drop=True)
    feats_needed = sorted({f for w, _ in channels.values() for f in w})
    for country, g in df.groupby("country", sort=False):
        q = g[q_mask[g.index]]
        p = g[g.src != 1]
        if len(q) == 0 or len(p) == 0:
            continue
        t0 = time.time()
        mats = {f: tfidf_hashed(hashed_matrix(q, f), hashed_matrix(p, f), max_df) for f in feats_needed}
        found = []
        for ch, (weights, k) in channels.items():
            Q, P = _combine(mats, weights, 0), _combine(mats, weights, 1)
            r, c, v = topk(Q, P, k)
            found.append(pd.DataFrame({"q": q.index.values[r], "p": p.index.values[c]}))
            if verbose:
                print(f"    [{country}] channel {ch}: {len(r):,} ({time.time() - t0:.0f}s)", flush=True)
        u = pd.concat(found).drop_duplicates().reset_index(drop=True)
        qi = np.searchsorted(q.index.values, u.q.values)
        pi = np.searchsorted(p.index.values, u.p.values)
        for ch, (weights, _) in channels.items():
            u["s_" + ch] = _rowwise_dot(_combine(mats, weights, 0), _combine(mats, weights, 1), qi, pi)
        out.append(u)
        if verbose:
            print(f"  [{country}] q={len(q):,} p={len(p):,} pairs={len(u):,} ({len(u) / len(q):.1f}/q) {time.time() - t0:.0f}s", flush=True)
        del mats
        gc.collect()
    return pd.concat(out, ignore_index=True)


def _rowwise_dot(Q, P, qi, pi, chunk=2_000_000):
    res = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        a = Q[qi[s:s + chunk]]
        b = P[pi[s:s + chunk]]
        res[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return res
