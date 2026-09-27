"""Residual-loss diagnostics of the final CE model (dev folds 1-4 only).

  1. S1 entities predicted empty under thr 0.9: how many are really non-singletons (score 0), and how
     good their top candidate is by probability band (what a top-1 fallback rule would gain or lose).
  2. Loss decomposition: per-S1 F0.5 lost, split by the kind of S1 error.
"""
import numpy as np
import pandas as pd

import config
import decision
from data_loader import read_ground_truth
from evaluation import evaluate_predictions
from prep import fold_of

TAG = "s3inv3n_ce"

if __name__ == "__main__":
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src"])
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    F["prob"] = np.load(config.work(f"oof_{TAG}.npy"))
    s1 = df.entity_id.values[(df.src == 1).values]
    ids = np.sort([x for x in s1 if fold_of(x) != 0])
    sel = decision.threshold_policy(F, 0.9)
    pred = decision.to_sets(df, sel, ids)
    r = evaluate_predictions(pred, gt, ids=ids, verbose=False, return_scores=True)
    sc = pd.Series(r["_scores"], index=ids)
    print(f"dev thr0.9 macro F0.5 {r['macro_f05']:.5f} | total loss {1 - r['macro_f05']:.5f} over {len(ids):,} S1")

    # --- 1. empty predictions
    n_true = pd.Series({i: len(gt.get(i, ())) for i in ids})
    empty = pd.Series({i: len(pred[i]) == 0 for i in ids})
    eid = df.entity_id.values
    best = F[F.fold.values != 0].sort_values("prob", ascending=False).drop_duplicates("q")
    best = best.assign(s1=eid[best.q.values]).set_index("s1")
    E = pd.DataFrame({"n_true": n_true[empty], "top_p": best.prob.reindex(n_true[empty].index).fillna(0),
                      "top_y": best.y.reindex(n_true[empty].index).fillna(0)})
    ns = E.n_true > 0
    print(f"\n[1] predicted empty: {empty.sum():,} ({empty.mean():.2%}) | true singletons among them {(~ns).sum():,} | "
          f"non-singletons {ns.sum():,} -> lost F0.5 {ns.sum() / len(ids):.5f}")
    E["band"] = pd.cut(E.top_p, [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
    t = E.groupby("band", observed=True).agg(S1=("n_true", "size"), singleton=("n_true", lambda v: (v == 0).sum()),
                                              top1_correct=("top_y", "sum"))
    t["top1_wrong_nonsingleton"] = t.S1 - t.singleton - t.top1_correct
    print("top candidate of empty-predicted S1, by probability band:\n" + t.to_string())

    # --- 2. loss decomposition
    n_pred = pd.Series({i: len(pred[i]) for i in ids})
    kind = np.select([(n_true == 0) & (n_pred > 0), (n_true > 0) & (n_pred == 0)],
                     ["singleton given matches", "non-singleton predicted empty"], "partial (missed/false links)")
    L = (1 - sc).groupby(kind).agg(["size", "sum"])
    L["loss_share_of_F05"] = L["sum"] / len(ids)
    print("\n[2] loss by S1 error kind:\n" + L.round(5).to_string())
