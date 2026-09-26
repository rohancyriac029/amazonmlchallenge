"""Categorise validation errors (dev folds only) and print representative examples.

  python error_analysis.py --tag T --policy P
"""
import argparse
import collections

import numpy as np
import pandas as pd

import config
import train
from data_loader import read_ground_truth
from prep import fold_of

COLS = ["q", "p", "y", "fold", "n_tset", "n_ratio", "n_skel_ratio", "a_tset", "a_first_num_eq",
        "p_a_empty", "p_n_indic", "p_n_is_domain", "p_n_has_dba", "q_f_core_s1", "s_combo"]


def categorise(r):
    """Heuristic error type for a (S1, other) pair row."""
    if r.p_a_empty == 1:
        base = "missing-address"
    elif r.n_tset >= 90 and r.a_tset < 70:
        base = "branch-ambiguity(same name, diff addr)"
    elif r.n_tset < 50 and r.a_tset >= 85:
        base = "address-only(DBA/renamed)"
    elif r.p_n_indic == 1:
        base = "transliteration"
    elif r.p_n_is_domain == 1:
        base = "domain-name"
    elif r.n_skel_ratio >= 90 and r.n_ratio < 90:
        base = "typo"
    elif r.a_tset < 85:
        base = "address-variation"
    else:
        base = "other"
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--policy", default="expF_miss0.0")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=COLS)
    F["prob"] = np.load(config.work(f"oof_{a.tag}.npy"))
    df = pd.read_parquet(config.work("train_v1.parquet"),
                         columns=["entity_id", "src", "business_name", "business_address", "country"])
    gt = read_ground_truth()
    sel = train.default_policies()[a.policy](F)
    ids = df.entity_id.values
    F["pred"] = 0
    F.loc[sel.index, "pred"] = 1
    F = F[F.fold != 0]                                    # dev only, never the holdout
    F["q_id"] = ids[F.q.values]
    F["p_id"] = ids[F.p.values]
    F["q_single"] = [len(gt[x]) == 0 for x in F.q_id]
    fp = F[(F.pred == 1) & (F.y == 0)].copy()
    fn = F[(F.pred == 0) & (F.y == 1)].copy()
    # blocking misses: true pairs never among candidates
    dev_s1 = [x for x in ids[(df.src == 1).values] if fold_of(x) != 0]
    cand = set(zip(F.q_id, F.p_id))
    n_block_miss = sum(1 for s in dev_s1 for o in gt[s] if (s, o) not in cand)
    print(f"dev S1={len(dev_s1):,} | FP pairs={len(fp):,} (on singletons {fp.q_single.sum():,}) | "
          f"FN pairs (rejected candidates)={len(fn):,} | blocking misses={n_block_miss:,}")
    rec = df.set_index("entity_id")
    for name, E in (("FALSE MERGES", fp), ("MISSED (rejected)", fn)):
        E["cat"] = [("singleton-FP:" if (name.startswith("FALSE") and s) else "") + categorise(r)
                    for s, r in zip(E.q_single, E.itertuples())]
        cnt = collections.Counter(E.cat)
        print(f"\n===== {name}: {len(E):,}")
        for c, n in cnt.most_common():
            print(f"  {n:8,}  {c}")
        for c, _ in cnt.most_common(6):
            ex = E[E.cat == c].sort_values("prob", ascending=not name.startswith("FALSE")).head(a.n)
            print(f"\n  --- {c}")
            for r in ex.itertuples():
                A, B = rec.loc[r.q_id], rec.loc[r.p_id]
                print(f"   p={r.prob:.3f} {r.q_id} | {A.business_name} | {A.business_address}")
                print(f"            {r.p_id} | {B.business_name} | {B.business_address}")


if __name__ == "__main__":
    main()
