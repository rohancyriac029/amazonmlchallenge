"""P1: stratified label-shift correction for near-copy candidates (second audit).

Within the near-copy stratum (name token-set >= 90 and address token-set >= 85), the
share of distractors pi can be estimated WITHOUT labels from two signatures whose
rates were measured on training data (audit2.py pi, validated: estimates 0.204 / 0.193
vs actual 0.193 on train):
    house-number differs:  distractor 0.899, true match 0.153
    extra name token:      distractor 0.472, true match 0.134
pi = mean of the two estimates.  Stratum pairs are re-weighted with the Bayes
label-shift correction  odds' = odds * [(1-pi_t)/(1-pi_r)] / [pi_t/pi_r],
per country (countries absent from the reference use the overall reference pi).
The dev-selected calibrated decision rule (expF_gate0.6) is then applied.
Everything above is fixed before scoring.

  python p1_shift.py validate     suite fold 4: clean vs injected
  python p1_shift.py test         write output/p1/ for the test set
"""
import os
import sys

import numpy as np
import pandas as pd

import config
import decision
import train
from data_loader import read_ground_truth
from evaluation import evaluate_predictions
from prep import fold_of

STRONG = dict(n_tset=90, a_tset=85)
R_NUM_D, R_NUM_T = 0.899, 0.153
R_TOK_D, R_TOK_T = 0.472, 0.134


def stratum(F):
    return (F.n_tset.values >= STRONG["n_tset"]) & (F.a_tset.values >= STRONG["a_tset"])


def estimate_pi(F, df, q_mask=None):
    """Label-free distractor share among near-copy records, per country (+ 'ALL')."""
    m = stratum(F) if q_mask is None else stratum(F) & q_mask
    s = F[m].sort_values("n_tset", ascending=False).drop_duplicates("p")
    a_num, b_num = df.a_first_num.values[s.q.values], df.a_first_num.values[s.p.values]
    both = (a_num != "") & (b_num != "")
    diff = both & (a_num != b_num)
    extra = np.array([len(set(y.split()) - set(x.split())) > 0
                      for x, y in zip(df.n_core.values[s.q.values], df.n_core.values[s.p.values])])
    cty = df.country.values[s.q.values]
    out = {}
    for c in ["ALL"] + sorted(np.unique(cty)):
        k = np.ones(len(s), bool) if c == "ALL" else cty == c
        p1 = (diff[k & both].mean() - R_NUM_T) / (R_NUM_D - R_NUM_T)
        p2 = (extra[k].mean() - R_TOK_T) / (R_TOK_D - R_TOK_T)
        out[c] = float(np.clip((p1 + p2) / 2, 0.01, 0.99))
    return out


def adjust(F, df, prob, pi_ref, pi_tgt):
    prob = np.clip(prob.astype(np.float64), 1e-7, 1 - 1e-7)
    out = prob.copy()
    m = stratum(F)
    cty = df.country.values[F.q.values]
    for c in np.unique(cty[m]):
        r = pi_ref.get(c, pi_ref["ALL"])
        t = pi_tgt.get(c, pi_tgt["ALL"])
        k = m & (cty == c)
        factor = ((1 - t) / (1 - r)) / (t / r)
        odds = prob[k] / (1 - prob[k]) * factor
        out[k] = odds / (1 + odds)
    return out.astype(np.float32)


COLS = ["q", "p", "n_tset", "a_tset"]
DF_COLS = ["entity_id", "src", "country", "n_core", "a_first_num"]


def validate():
    gt = read_ground_truth()
    Ftr = pd.read_parquet(config.work("train_feat.parquet"), columns=COLS + ["fold"])
    dtr = pd.read_parquet(config.work("train_v1.parquet"), columns=DF_COLS)
    ref_mask = (Ftr.fold.values != 4) & (Ftr.fold.values != 0)
    pi_ref = estimate_pi(Ftr, dtr, ref_mask)
    print("reference pi (folds 1-3):", {k: round(v, 3) for k, v in pi_ref.items()})
    Fs = pd.read_parquet(config.work("train_sim_feat.parquet"), columns=COLS)
    ds = pd.read_parquet(config.work("train_sim_v1.parquet"), columns=DF_COLS)
    s1 = dtr[(dtr.src == 1).values]
    ids = np.sort([x for x in s1.entity_id.values if fold_of(x) == 4])
    for name, F, d, prob in (("clean", Ftr, dtr, np.load(config.work("oof_s3inv3.npy"))),
                             ("injected", Fs, ds, np.load(config.work("sim_prob_v3.npy")))):
        is4 = np.zeros(len(d), bool)
        is4[np.flatnonzero(d.entity_id.isin(set(ids)).values)] = True
        pi_t = estimate_pi(F, d, is4[F.q.values])
        print(f"\n[{name}] target pi (fold 4):", {k: round(v, 3) for k, v in pi_t.items()})
        adj = adjust(F, d, prob, pi_ref, pi_t)
        for label, pr, pol in (("raw expF_gate0.6", prob, "expF_gate0.6"), ("raw thr0.9", prob, "thr0.9"),
                               ("P1 + expF_gate0.6", adj, "expF_gate0.6"), ("P1 + thr0.9", adj, "thr0.9")):
            X = F[["q", "p"]].copy(); X["prob"] = pr
            pred = decision.to_sets(d, train.default_policies()[pol](X), ids)
            print(f"  {label:20s}", end=" ")
            r = evaluate_predictions(pred, gt, ids=ids)
            train.log_experiment({"tag": f"P1_{label}", "eval": f"fold4_{name}", "note": "second-audit P1",
                                  **{k: v for k, v in r.items() if not k.startswith("_")}})


def test():
    from submission import write_sets, self_check
    Ftr = pd.read_parquet(config.work("train_feat.parquet"), columns=COLS + ["fold"])
    dtr = pd.read_parquet(config.work("train_v1.parquet"), columns=DF_COLS)
    pi_ref = estimate_pi(Ftr, dtr, Ftr.fold.values != 0)
    Ft = pd.read_parquet(config.work("test_feat.parquet"), columns=COLS)
    dt = pd.read_parquet(config.work("test_v1.parquet"), columns=DF_COLS)
    pi_t = estimate_pi(Ft, dt)
    print("reference pi:", {k: round(v, 3) for k, v in pi_ref.items()}, "| test pi:", {k: round(v, 3) for k, v in pi_t.items()})
    X = Ft[["q", "p"]].copy()
    X["prob"] = adjust(Ft, dt, np.load(config.work("test_prob_s3inv3.npy")), pi_ref, pi_t)
    s1 = dt.entity_id.values[(dt.src == 1).values]
    preds = decision.to_sets(dt, train.default_policies()["expF_gate0.6"](X), s1)
    out = os.path.join(config.OUT_DIR, "p1")
    mp_, cp_ = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    write_sets(mp_, s1, preds, "matched_entity_ids")
    write_sets(cp_, s1, decision.to_sets(dt, X, s1), "candidate_entity_ids")
    self_check(mp_, cp_, s1, dt.entity_id.values[(dt.src != 1).values])
    n = pd.Series([len(preds[i]) for i in s1]); c = dt.set_index("entity_id").country.reindex(s1).values
    print("matches/S1:", n.groupby(c).mean().round(3).to_dict(), "| empty:", (n == 0).groupby(c).mean().round(4).to_dict())


if __name__ == "__main__":
    {"validate": validate, "test": test}[sys.argv[1]]()
