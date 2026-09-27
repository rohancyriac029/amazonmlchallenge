"""Cross-encoder (CE) stage, step 1: choose the pairs the CE scores and export their entity ids.

The CE only scores pairs whose stage-2 probability is not already decisive, selected by a
rule that is computed identically on train (stage-2 OOF) and test (full stage-2 model):
p2 >= P2_MIN and within the top TOPK candidates of its S1 by p2.

  python ce_export.py diag      error coverage / size of candidate subset rules (dev folds only)
  python ce_export.py export    -> work/ce_pairs_{train,test}.parquet  (row, p2, entity ids; train also fold, y)
  python ce_export.py p3        -> work/ce_pairs_test_p3.npy  final stage-3 test probability per exported pair
                                   (pseudo-labels for ce_train.py adapt)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import final2

S2, TAG = "inv3n", "s3inv3n"
P2_MIN, TOPK = 0.01, 8


def test_p2():
    path = config.work(f"test_p2_{TAG}.npy")
    try:
        return np.load(path)
    except FileNotFoundError:
        spec = final2.s2_spec(S2)
        final2.ensure_test_features(spec)
        X, _, _ = final2.matrix("test_feat", spec)
        p2 = lgb.Booster(model_file=config.work(f"stage2_{TAG}.txt")).predict(X, num_threads=config.N_JOBS).astype(np.float32)
        np.save(path, p2)
        return p2


def select(q, p2, p2_min=P2_MIN, topk=TOPK):
    rank = pd.DataFrame({"q": q, "p2": p2}).groupby("q").p2.rank(ascending=False, method="first").values
    return (p2 >= p2_min) & (rank <= topk)


def diag():
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    p2 = np.load(config.work(f"goof_{S2}__normal.npy"))
    p3 = np.load(config.work(f"oof_{TAG}.npy"))
    dev = F.fold.values != 0
    y = F.y.values == 1
    acc = (p3 >= 0.9) & (p3 >= pd.Series(p3).groupby(F.p.values).transform("max").values)
    fn, fp = dev & y & ~acc, dev & ~y & acc
    n_s1 = F.q[dev].nunique()
    print(f"dev: {dev.sum():,} pairs, {n_s1:,} S1 | stage-3 thr0.9 FN {fn.sum():,} FP {fp.sum():,}")
    tp2 = test_p2()
    Tq = pd.read_parquet(config.work("test_feat.parquet"), columns=["q"]).q.values
    for p2_min in (0.003, 0.01, 0.03):
        for topk in (3, 5, 8):
            m = select(F.q.values, p2, p2_min, topk)
            mt = select(Tq, tp2, p2_min, topk)
            print(f"p2>={p2_min:<5} top{topk}: dev pairs {(m & dev).sum():>10,} ({(m & dev).sum() / n_s1:.2f}/S1) "
                  f"covers FN {(m & fn).sum() / fn.sum():.3f} FP {(m & fp).sum() / fp.sum():.3f} | "
                  f"all train {m.sum():,} test {mt.sum():,} ({mt.sum() / len(np.unique(Tq)):.2f}/S1)")


def export():
    """Only ids are exported: the text is read from the raw TSVs where the CE runs."""
    for split in ("train", "test"):
        F = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=["q", "p"] + (["y", "fold"] if split == "train" else []))
        p2 = np.load(config.work(f"goof_{S2}__normal.npy")) if split == "train" else test_p2()
        rows = np.flatnonzero(select(F.q.values, p2))
        eid = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["entity_id"]).entity_id.values
        out = pd.DataFrame({"row": rows.astype(np.int64), "p2": p2[rows].astype(np.float32),
                            "q_id": eid[F.q.values[rows]], "p_id": eid[F.p.values[rows]]})
        if split == "train":
            out["fold"] = F.fold.values[rows].astype(np.int8)
            out["y"] = F.y.values[rows].astype(np.int8)
            out["p3"] = np.load(config.work(f"oof_{TAG}.npy"))[rows].astype(np.float32)   # diagnostics only
        out.to_parquet(config.work(f"ce_pairs_{split}.parquet"), index=False, compression="zstd")
        print(f"{split}: {len(out):,} pairs exported")


def p3(tag="s3inv3n_ce"):
    rows = pd.read_parquet(config.work("ce_pairs_test.parquet"), columns=["row"]).row.values
    np.save(config.work("ce_pairs_test_p3.npy"), np.load(config.work(f"test_prob_{tag}.npy"))[rows].astype(np.float32))


if __name__ == "__main__":
    {"diag": diag, "export": export, "p3": p3}[sys.argv[1]]()
