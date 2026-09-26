"""Label-shift (prior-shift) correction of pair probabilities, Saerens et al. (2002).

The matcher is calibrated to the training share of true pairs among candidates.
If a new dataset contains a different share (e.g. more near-copy distractors),
EM on the model's own posteriors estimates the new prior without labels, and
posteriors are re-weighted:  p' = a p / (a p + b (1 - p)),  a = pi'/pi, b = (1-pi')/(1-pi).
Estimated per country (open set), so unseen countries get their own correction.

  python prior_shift.py validate --tag stack1       offline check on dev folds (never the holdout)
"""
import argparse

import numpy as np
import pandas as pd

import config


def em_prior(p, pi_train, iters=200, tol=1e-7):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    pi = pi_train
    for _ in range(iters):
        a, b = pi / pi_train, (1 - pi) / (1 - pi_train)
        post = a * p / (a * p + b * (1 - p))
        new = post.mean()
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi


def adjust(p, pi_train, pi_new):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    a, b = pi_new / pi_train, (1 - pi_new) / (1 - pi_train)
    return (a * p / (a * p + b * (1 - p))).astype(np.float32)


def correct_by_group(prob, groups, pi_train):
    """Estimate and apply the prior shift separately for each group (country)."""
    out = prob.copy()
    est = {}
    for g in np.unique(groups):
        m = groups == g
        pi = em_prior(prob[m], pi_train)
        est[g] = pi
        out[m] = adjust(prob[m], pi_train, pi)
    return out, est


def validate(a):
    import decision
    import train
    from data_loader import read_ground_truth
    from evaluation import evaluate_predictions, paired_bootstrap
    from prep import fold_of
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country"])
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    F["prob"] = np.load(config.work(f"oof_{a.tag}.npy"))
    dev = F[F.fold != 0].reset_index(drop=True)
    pi_train = float(dev.y.mean())
    print(f"training prior pi={pi_train:.4f}; mean OOF prob {dev.prob.mean():.4f} (calibration check)")
    ids_all = df.entity_id.values
    s1 = ids_all[(df.src == 1).values]
    dev_ids = np.sort([x for x in s1 if fold_of(x) != 0])
    cty = df.country.values[dev.q.values]
    owned = np.flatnonzero(dev.y.values == 1)
    rng = np.random.default_rng(0)
    pol = train.default_policies()[a.policy]
    for rate in (0.0, 0.2, 0.35):
        # hide a random share of true-match records -> higher distractor share among candidates
        pool_true = np.unique(dev.p.values[owned])
        hide = set(rng.choice(pool_true, int(len(pool_true) * rate), replace=False)) if rate else set()
        keep = ~np.isin(dev.p.values, list(hide)) if hide else np.ones(len(dev), bool)
        D = dev[keep].reset_index(drop=True)
        hide_ids = set(ids_all[list(hide)]) if hide else set()
        g = {k: (v - hide_ids) for k, v in gt.items()} if hide else gt
        c = cty[keep]
        adj, est = correct_by_group(D.prob.values, c, pi_train)
        true_pi = {k: float(D.y.values[c == k].mean()) for k in np.unique(c)}
        print(f"\n== hidden true records {rate:.0%}: true prior {true_pi} | EM estimate { {k: round(v, 4) for k, v in est.items()} }")
        res = {}
        for name, pr in (("raw", D.prob.values), ("em", adj)):
            X = D[["q", "p"]].copy(); X["prob"] = pr
            pred = decision.to_sets(df, pol(X), dev_ids)
            print(f"  {name:4s}", end=" ")
            res[name] = evaluate_predictions(pred, g, ids=dev_ids, return_scores=True)
        d, lo, hi = paired_bootstrap(res["raw"]["_scores"], res["em"]["_scores"])
        print(f"  EM - raw: {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate"])
    ap.add_argument("--tag", default="stack1")
    ap.add_argument("--policy", default="expF_gate0.6")
    validate(ap.parse_args())
