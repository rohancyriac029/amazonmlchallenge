"""Compare stage-3 OOF runs on dev folds 1-4 (never the holdout).

  python e_compare.py pair  <tagA> <tagB>          bootstrap + per-class FN/FP deltas
  python e_compare.py seeds <tag1> <tag2> ...      seed std of dev macro F0.5
"""
import sys

import numpy as np
import pandas as pd

import config
import train
import decision
from data_loader import read_ground_truth
from evaluation import evaluate_predictions, paired_bootstrap
from prep import fold_of

POLS = ("thr0.9", "expF_gate0.6")
CLS_COLS = ["q", "p", "y", "fold", "n_tset", "a_tset", "p_a_empty", "a_first_num_eq", "a_nn_q", "a_nn_p",
            "p_n_indic", "n_skel_ratio", "n_ratio"]


def classify(X):
    bn = (X.a_nn_q > 0) & (X.a_nn_p > 0)
    return np.select([X.p_a_empty == 1, X.n_tset < 50, X.p_n_indic == 1,
                      bn & (X.a_first_num_eq == 0) & (X.n_tset >= 90),
                      (X.n_skel_ratio >= 90) & (X.n_ratio < 90), X.a_tset < 70],
                     ["no address", "name different (DBA)", "Indic name", "same name, house no. differs",
                      "typo in name", "address differs"], "other")


def _setup():
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src"])
    s1 = df.entity_id.values[(df.src == 1).values]
    ids = np.sort([x for x in s1 if fold_of(x) != 0])
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=CLS_COLS)
    return gt, df, ids, F


def score(tag, gt, df, ids, F, pol):
    X = F[["q", "p"]].copy(); X["prob"] = np.load(config.work(f"oof_{tag}.npy"))
    sel = train.default_policies()[pol](X)
    r = evaluate_predictions(decision.to_sets(df, sel, ids), gt, ids=ids, verbose=False, return_scores=True)
    acc = np.zeros(len(F), bool); acc[sel.index.values] = True
    return r, acc


def pair(a, b):
    gt, df, ids, F = _setup()
    dev = F.fold.values != 0
    cls = pd.Series(classify(F), index=F.index)
    for pol in POLS:
        ra, acc_a = score(a, gt, df, ids, F, pol)
        rb, acc_b = score(b, gt, df, ids, F, pol)
        d, lo, hi = paired_bootstrap(ra["_scores"], rb["_scores"])
        print(f"\n[{pol}] {a}: {ra['macro_f05']:.5f}  {b}: {rb['macro_f05']:.5f}  delta {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  "
              f"FP {ra['false_merges']:,}->{rb['false_merges']:,}  missed {ra['missed']:,}->{rb['missed']:,}")
        y = F.y.values == 1
        rows = []
        for c in sorted(cls.unique()):
            m = dev & (cls.values == c)
            rows.append({"class": c, f"FN {a}": int((m & y & ~acc_a).sum()), f"FN {b}": int((m & y & ~acc_b).sum()),
                         f"FP {a}": int((m & ~y & acc_a).sum()), f"FP {b}": int((m & ~y & acc_b).sum())})
        print(pd.DataFrame(rows).set_index("class").to_string())
        train.log_experiment({"tag": f"cmp_{a}_vs_{b}", "policy": pol, "delta": d, "ci": [lo, hi],
                              "f_a": ra["macro_f05"], "f_b": rb["macro_f05"]})


def seeds(tags):
    gt, df, ids, F = _setup()
    for pol in POLS:
        v = [score(t, gt, df, ids, F, pol)[0]["macro_f05"] for t in tags]
        print(f"[{pol}] " + "  ".join(f"{t}={x:.5f}" for t, x in zip(tags, v))
              + f"  | mean {np.mean(v):.5f}  seed std {np.std(v, ddof=1):.5f}  -> 2x std = {2 * np.std(v, ddof=1):.5f}")
        train.log_experiment({"tag": "E9_seed_std", "policy": pol, "runs": tags, "scores": v,
                              "seed_std": float(np.std(v, ddof=1))})


if __name__ == "__main__":
    {"pair": lambda: pair(sys.argv[2], sys.argv[3]), "seeds": lambda: seeds(sys.argv[2:])}[sys.argv[1]]()
