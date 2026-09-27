"""Unseen-country simulation for cross-encoder self-training (proxy for France, absent from training).

India plays the unseen country: stage 2 is the US-only model of `exp.py --loco US:India` (tag
lo_usin_a8n), the cross-encoder is fine-tuned on US pairs only, and a logistic combiner of
[logit p2, CE logit] fitted on held-out US pairs stands in for stage 3.  Adaptation = continue
fine-tuning the US cross-encoder on confident India predictions (pseudo-labels; no India labels),
mixed with US labelled pairs, then rescore India.  India labels are used only for evaluation,
on dev folds 1-4 (the holdout fold is excluded throughout).

Pre-registered gate: India dev macro F0.5 (thr 0.9 + exclusivity) must rise by >= +0.002 with a
paired-bootstrap 95% CI above 0; only then is the method applied to France.

  python ce_loco.py export   (CPU box with the feature tables) -> work/loco_in_pairs.parquet, loco_in_s1.parquet
  python ce_loco.py run      (GPU)                             -> report + work/ce_loco_*.{pt,npy}
"""
import os
import sys

import numpy as np
import pandas as pd

import config

S2LOCO = "lo_usin_a8n"
P2_MIN, P2_MAX, TOPK = 0.01, 0.999, 8            # same pair rule as ce_export / ce_train
PSEUDO_HI, PSEUDO_LO = 0.97, 0.03                 # as adapt.py
N_US_TRAIN, N_ADAPT, N_COMB, ADAPT_LR = 600_000, 300_000, 300_000, 2e-5


def export():
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    p2 = np.load(config.work(f"goof_{S2LOCO}__normal.npy"))
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "country", "src"])
    eid = df.entity_id.values
    m = (df.country.values[F.q.values] == "India") & (F.fold.values != 0)
    rank = pd.DataFrame({"q": F.q.values[m], "p2": p2[m]}).groupby("q").p2.rank(ascending=False, method="first").values
    keep = p2[m] >= P2_MIN
    rows = np.flatnonzero(m)[keep]
    out = pd.DataFrame({"q_id": eid[F.q.values[rows]], "p_id": eid[F.p.values[rows]], "p2": p2[rows].astype(np.float32),
                        "y": F.y.values[rows].astype(np.int8), "fold": F.fold.values[rows].astype(np.int8),
                        "region": (rank[keep] <= TOPK) & (p2[rows] < P2_MAX)})
    out.to_parquet(config.work("loco_in_pairs.parquet"), index=False, compression="zstd")
    from prep import fold_of
    s1 = eid[(df.src.values == 1) & (df.country.values == "India")]
    pd.DataFrame({"entity_id": [x for x in s1 if fold_of(x) != 0]}).to_parquet(config.work("loco_in_s1.parquet"), index=False)
    print(f"India dev pairs with p2 >= {P2_MIN}: {len(out):,} (CE region {out.region.sum():,})")


def run():
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import ce_train as C
    import decision
    from data_loader import read_ground_truth
    from evaluation import evaluate_predictions, paired_bootstrap

    rng = np.random.default_rng(0)
    cty = pd.read_csv(config.src_path("train", 1), sep="\t", dtype=str, usecols=["entity_id", "country"]).set_index("entity_id").country
    U = pd.read_parquet(config.work("ce_pairs_train.parquet"), columns=["q_id", "p_id", "p2", "fold", "y"])
    U = U[(U.p2.values < P2_MAX) & (U.fold.values != 0)]
    U = U[cty.reindex(U.q_id.values).values == "US"].reset_index(drop=True)
    I = pd.read_parquet(config.work("loco_in_pairs.parquet"))
    s1_in = pd.read_parquet(config.work("loco_in_s1.parquet")).entity_id.values
    ids = pd.Index(pd.unique(np.concatenate([U.q_id.values, U.p_id.values, I.q_id.values, I.p_id.values, s1_in])))
    text = C.texts("train", ids)
    for D in (U, I):
        D["qi"] = ids.get_indexer(D.q_id.values).astype(np.int32)
        D["pi"] = ids.get_indexer(D.p_id.values).astype(np.int32)
    print(f"US pairs {len(U):,} | India pairs {len(I):,} (region {I.region.sum():,}) | India dev S1 {len(s1_in):,}", flush=True)

    reg = np.flatnonzero(I.region.values)
    u4 = rng.choice(np.flatnonzero(U.fold.values == 4), N_COMB, replace=False)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    gt = read_ground_truth()
    dfx = pd.DataFrame({"entity_id": ids.values})

    def evaluate(name, weights):
        cache = config.work(f"ce_loco_{name}.npz")
        if os.path.exists(cache):
            z = np.load(cache); ce_u4, ce_in = z["u4"], z["ind"]
        else:
            tok = C.AutoTokenizer.from_pretrained(C.MODEL)
            model = C.CrossEncoder().to(C.DEV)
            model.load_state_dict(torch.load(weights, map_location=C.DEV))
            ce_u4 = C.score(model, tok, text, U.qi.values[u4], U.pi.values[u4])
            ce_in = C.score(model, tok, text, I.qi.values[reg], I.pi.values[reg])
            np.savez(cache, u4=ce_u4, ind=ce_in)
        comb = LogisticRegression(C=1.0).fit(np.c_[lg(U.p2.values[u4]), ce_u4], U.y.values[u4])
        prob = I.p2.values.astype(np.float64).copy()
        prob[reg] = comb.predict_proba(np.c_[lg(I.p2.values[reg]), ce_in])[:, 1]
        F = pd.DataFrame({"q": I.qi.values, "p": I.pi.values, "prob": prob})
        r = evaluate_predictions(decision.to_sets(dfx, decision.threshold_policy(F, 0.9), s1_in), gt, ids=s1_in,
                                 verbose=False, return_scores=True)
        yr = I.y.values[reg]
        print(f"[{name}] India CE AUC {roc_auc_score(yr, ce_in):.4f} | combined AUC {roc_auc_score(yr, prob[reg]):.4f} | "
              f"India dev macro F0.5 {r['macro_f05']:.5f} P {r['macro_precision']:.4f} R {r['macro_recall']:.4f} "
              f"| combiner coefs {comb.coef_.round(3).tolist()}", flush=True)
        return prob, r

    # 1. US-only cross-encoder (folds 1-3; fold 4 fits the combiner)
    w_us = config.work("ce_loco_us.pt")
    if not os.path.exists(w_us):
        model, _ = C.fit(U, text, C.sample_train(U, (1, 2, 3), N_US_TRAIN, 0))
        torch.save(model.state_dict(), w_us)
    prob0, r0 = evaluate("us", w_us)
    p2_only = evaluate_predictions(decision.to_sets(dfx, decision.threshold_policy(
        pd.DataFrame({"q": I.qi.values, "p": I.pi.values, "prob": I.p2.values}), 0.9), s1_in), gt, ids=s1_in, verbose=False)
    print(f"reference: US-only stage 2 alone, India dev macro F0.5 {p2_only['macro_f05']:.5f}", flush=True)

    # 2. self-training on confident India predictions (pseudo-labels), mixed with US labelled pairs
    w_ad = config.work("ce_loco_adapted.pt")
    if not os.path.exists(w_ad):
        pr = prob0[reg]
        pos, neg = reg[pr >= PSEUDO_HI], reg[pr <= PSEUDO_LO]
        yt = I.y.values
        print(f"pseudo-labels: {len(pos):,} pos (precision {yt[pos].mean():.4f}), {len(neg):,} neg "
              f"(purity {1 - yt[neg].mean():.4f}); uncertain left out {len(reg) - len(pos) - len(neg):,}", flush=True)
        pick = rng.choice(np.r_[pos, neg], min(N_ADAPT, len(pos) + len(neg)), replace=False)
        us = C.sample_train(U, (1, 2, 3), N_ADAPT, 1)
        A = pd.DataFrame({"qi": np.r_[I.qi.values[pick], U.qi.values[us]], "pi": np.r_[I.pi.values[pick], U.pi.values[us]],
                          "y": np.r_[(prob0[pick] >= PSEUDO_HI).astype(np.int8), U.y.values[us]]})
        model, _ = C.fit(A, text, rng.permutation(len(A)), 2, init=w_us, lr=ADAPT_LR)
        torch.save(model.state_dict(), w_ad)
    _, r1 = evaluate("adapted", w_ad)

    d, lo, hi = paired_bootstrap(r0["_scores"], r1["_scores"])
    print(f"\nadapted - US-only: India dev macro F0.5 {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    print("GATE", "PASS" if d >= 0.002 and lo > 0 else "FAIL", "(needs >= +0.002 and CI above 0)")


if __name__ == "__main__":
    {"export": export, "run": run}[sys.argv[1]]()
