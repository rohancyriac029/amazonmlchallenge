"""Change A: country-relative IDF-weighted similarities (label-free, transfer across countries).

Token weights come from each country's own records in the split being scored
(unlabeled), normalised to [0, 1] by log(N_country) so that countries of
different sizes are on the same scale.  Common words of *any* language
("street", "rue", "llc", "sarl", "nagar") are down-weighted automatically,
with no hand-written lists.

  python features_idf.py <split> <feat_table>   e.g. train train_feat / train train_dense_feat / test test_feat
  -> work/<feat_table>_A.parquet  (row-aligned with the feature table)
"""
import math
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

import config

_A = {}
A_FEATS = ["n_wjacc", "n_wcov_q", "n_wcov_p", "n_soft", "n_miss_maxw",
           "a_wjacc", "a_wcov_q", "a_wcov_p", "a_soft", "a_miss_maxw",
           "q_f_core_s1_rel", "p_f_core_pool_rel", "q_f_core_num_s1_rel", "p_f_addr_pool_rel"]


def country_idf(tokens_by_record, country):
    """{country: {token: normalised idf}}, plus a default weight for unseen tokens."""
    out = {}
    for c in np.unique(country):
        idx = np.flatnonzero(country == c)
        dfc = Counter()
        for i in idx:
            dfc.update(set(tokens_by_record[i].split()))
        n = len(idx)
        ln = math.log(n + 1)
        out[c] = {t: math.log((n + 1) / (v + 1)) / ln for t, v in dfc.items()}
    return out


def _weighted(a, b, w):
    if not a or not b:
        return [-1.0] * 5
    wa = {t: w.get(t, 1.0) for t in a}
    wb = {t: w.get(t, 1.0) for t in b}
    shared = a & b
    ws = sum(wa[t] for t in shared)
    sa, sb = sum(wa.values()) or 1e-9, sum(wb.values()) or 1e-9
    union = sa + sb - ws
    # soft TF-IDF: each token matched to its best counterpart if Jaro-Winkler >= 0.88
    def soft(x, y, wx):
        tot = 0.0
        for t in x:
            best = 1.0 if t in y else max((JaroWinkler.normalized_similarity(t, u) for u in y), default=0.0)
            if best >= 0.88:
                tot += wx[t] * best
        return tot / (sum(wx.values()) or 1e-9)
    s = 0.5 * (soft(a, b, wa) + soft(b, a, wb))
    unmatched = [wa[t] for t in a - b] + [wb[t] for t in b - a]
    return [ws / union, ws / sa, ws / sb, s, max(unmatched, default=0.0)]


def _block(bounds):
    s, e = bounds
    q, p, cty = _A["q"][s:e], _A["p"][s:e], _A["cty"][s:e]
    name, addr, wn, wa = _A["name"], _A["addr"], _A["wn"], _A["wa"]
    out = np.empty((e - s, 10), dtype=np.float32)
    for i, (qi, pi, c) in enumerate(zip(q, p, cty)):
        out[i, :5] = _weighted(set(name[qi].split()), set(name[pi].split()), wn[c])
        out[i, 5:] = _weighted(set(addr[qi].split()), set(addr[pi].split()), wa[c])
    return out


def build(split, table, chunk=20000):
    t0 = time.time()
    df = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["entity_id", "src", "country", "n_core", "a_words"])
    cty_rec = df.country.values
    visible = np.ones(len(df), bool)
    if "dense" in table:          # hidden S1 records do not exist in the simulated split
        import dense
        visible = ~dense.dropped_s1(df)
    if "univ" in table:           # removed entities (S1 + their records) do not exist
        import dense
        visible = ~dense.universe_removed(df)
    cty_idf = np.where(visible, cty_rec, "__hidden__")
    wn = country_idf(df.n_core.tolist(), cty_idf)
    wa = country_idf(df.a_words.tolist(), cty_idf)
    print(f"idf tables built ({time.time() - t0:.0f}s)", flush=True)
    F = pd.read_parquet(config.work(f"{table}.parquet"),
                        columns=["q", "p", "q_f_core_s1", "p_f_core_pool", "q_f_core_num_s1", "p_f_addr_pool"])
    shared = {"q": F.q.values, "p": F.p.values, "cty": cty_rec[F.q.values],
              "name": df.n_core.tolist(), "addr": df.a_words.tolist(), "wn": wn, "wa": wa}
    bounds = [(s, min(s + chunk, len(F))) for s in range(0, len(F), chunk)]
    X = np.vstack(config.parallel_map(_block, bounds, "_A", shared))
    out = pd.DataFrame(X, columns=A_FEATS[:10])
    # relative ambiguity statistics: counts per 100k records of the same country and source group
    n_s1 = pd.Series(cty_rec[(df.src == 1).values & visible]).value_counts()
    n_pool = pd.Series(cty_rec[(df.src != 1).values & visible]).value_counts()
    cq = pd.Series(cty_rec[F.q.values])
    s1n = cq.map(n_s1).values.astype(np.float32)
    pln = cq.map(n_pool).values.astype(np.float32)
    out["q_f_core_s1_rel"] = F.q_f_core_s1.values / s1n * 1e5
    out["p_f_core_pool_rel"] = F.p_f_core_pool.values / pln * 1e5
    out["q_f_core_num_s1_rel"] = F.q_f_core_num_s1.values / s1n * 1e5
    out["p_f_addr_pool_rel"] = np.where(F.p_f_addr_pool.values < 0, -1, F.p_f_addr_pool.values / pln * 1e5)
    out.astype(np.float32).to_parquet(config.work(f"{table}_A.parquet"), index=False)
    print(f"{table}_A {out.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
