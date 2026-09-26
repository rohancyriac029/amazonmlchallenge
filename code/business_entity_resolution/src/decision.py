"""Turn pair probabilities into per-S1 match sets.

Policies
  threshold : keep pairs with prob >= t
  expected_f: per S1, pick the top-k prefix (k >= 0) maximising a plug-in
              estimate of expected F0.5, where the empty set scores
              P(no match) and the expected number of true matches is the sum
              of candidate probabilities (optionally inflated for misses).
Exclusivity: ground truth shows each S2/S3 record belongs to at most one S1,
so a pool record is only kept for the S1 where it scores highest.
"""
import numpy as np
import pandas as pd


def exclusive(pairs, prob_col="prob"):
    """Keep only the best-scoring S1 for every pool record."""
    best = pairs.groupby("p")[prob_col].transform("max")
    return pairs[pairs[prob_col] >= best]


def threshold_policy(pairs, t, prob_col="prob", excl=True):
    x = pairs[pairs[prob_col] >= t]
    if excl:
        x = exclusive(x, prob_col)
    return x


def expected_f_policy(pairs, prob_col="prob", excl=True, min_prob=0.05, miss_mass=0.0, t_floor=0.0):
    """Vectorised per-S1 prefix selection by expected F0.5 (plug-in)."""
    x = pairs[pairs[prob_col] >= min_prob]
    if excl:
        x = exclusive(x, prob_col)
    x = x.sort_values(["q", prob_col], ascending=[True, False])
    q = x.q.values
    p = x[prob_col].values.astype(np.float64)
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(q)]))
    csum = np.cumsum(p)
    base = np.r_[0.0, csum][starts][grp]
    tp_k = csum - base                              # expected TP of prefix
    k = np.arange(len(q)) - starts[grp] + 1         # prefix size
    tot = pd.Series(p).groupby(grp).transform("sum").values + miss_mass
    ef = 1.25 * tp_k / (k + 0.25 * tot)
    # empty set: probability that the S1 has no match ~ prod(1 - p)
    log1m = np.log1p(-np.clip(p, 0, 1 - 1e-9))
    p_empty = np.exp(pd.Series(log1m).groupby(grp).transform("sum").values)
    best_ef = pd.Series(ef).groupby(grp).transform("max").values
    # choose argmax prefix per group
    is_best = ef >= best_ef - 1e-12
    first_best_k = pd.Series(np.where(is_best, k, 10 ** 9)).groupby(grp).transform("min").values
    keep = (k <= first_best_k) & (best_ef > p_empty) & (p >= t_floor)
    return x[keep]


def to_sets(df, sel, q_ids):
    """sel: pairs with q, p row indices -> dict S1 id -> set of ids."""
    ids = df.entity_id.values
    out = {i: set() for i in q_ids}
    for a, b in zip(ids[sel.q.values], ids[sel.p.values]):
        t = out.get(a)
        if t is not None:          # pairs of S1 outside q_ids are ignored
            t.add(b)
    return out


def gated_expected_f_policy(pairs, gate=0.6, prob_col="prob", **kw):
    """Expected-F selection, but predict the empty set unless the best candidate
    of the S1 reaches `gate` (singleton guard)."""
    sel = expected_f_policy(pairs, prob_col=prob_col, **kw)
    mx = pairs.groupby("q")[prob_col].max()
    return sel[sel.q.map(mx).values >= gate]
