"""CE memorisation check: is the dev gain real on records the CE never trained on?

Folds split the S1 entities, but an S2/S3 record can appear in a CE training pair (as the
match or a distractor of a training-fold S1) and again, as a candidate of a scored-fold S1.
The test set shares no records with train, so only the gain on *unseen* records transfers.
For each dev pair the CE scored, flag whether its S2/S3 record was in the training sample
of the half-model that scored it, then compare the stage-3 pair errors (thr 0.9) with and
without the CE inside each group.

Re-draws the training samples exactly as ce_train.sample_train does (same filter, seeds, sizes).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

P2_MAX, N_TRAIN = 0.999, 1_000_000          # as in ce_train.py
HALVES = {"A": ((1, 2), (3, 4), 0), "B": ((3, 4), (1, 2), 1)}


def sample_train(P, folds, n, seed):          # mirror of ce_train.sample_train
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(P.fold.isin(folds).values)
    unc = cand[(P.p2.values[cand] > 0.02) & (P.p2.values[cand] < 0.98)]
    rest = np.setdiff1d(cand, unc)
    n_unc = min(len(unc), int(n * 0.7))
    return np.concatenate([rng.choice(unc, n_unc, replace=False), rng.choice(rest, n - n_unc, replace=False)])


if __name__ == "__main__":
    P = pd.read_parquet(config.work("ce_pairs_train.parquet"), columns=["row", "p2", "p_id", "fold", "y"])
    P["ce"] = np.load(config.work("ce_logit_train.npy"))
    P = P[P.p2.values < P2_MAX].reset_index(drop=True)
    assert P.ce.notna().all()
    base, new = np.load(config.work("oof_s3inv3n.npy")), np.load(config.work("oof_s3inv3n_ce.npy"))
    P["b"], P["n"] = base[P.row.values], new[P.row.values]
    rows = []
    for name, (trf, scf, seed) in HALVES.items():
        seen_ids = set(P.p_id.values[sample_train(P, trf, N_TRAIN, seed)])
        S = P[P.fold.isin(scf).values]
        seen = S.p_id.isin(seen_ids).values
        for g, m in (("seen in CE training", seen), ("never seen", ~seen)):
            X = S[m]
            y = X.y.values == 1
            eb = int(((y & (X.b.values < 0.9)) | (~y & (X.b.values >= 0.9))).sum())
            en = int(((y & (X.n.values < 0.9)) | (~y & (X.n.values >= 0.9))).sum())
            rows.append({"model": name, "record": g, "pairs": len(X), "pos_rate": round(y.mean(), 3),
                         "CE AUC": round(roc_auc_score(y, X.ce.values), 4),
                         "stage-3 AUC": round(roc_auc_score(y, X.b.values), 4),
                         "errors before": eb, "errors with CE": en, "change": f"{(en - eb) / eb:+.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))
