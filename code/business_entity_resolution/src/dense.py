"""S1-dropout: simulate the test set's higher distractor density on training data.

Test has 5.75 S2/S3 records per S1 vs 4.68 in train (about 2x the unowned
distractors). Hiding a random fraction of S1 entities turns their S2/S3 records
into unowned distractors for all remaining S1s, which reproduces that density
without inventing data. Retrieval per S1 is unchanged (the pool is unchanged),
so we filter the candidate table, recompute every cross-S1 (context /
competition) feature and the stage-1 ranker on the denser set, reuse the pure
pairwise features and compute them only for pairs that newly enter the top-25.

  python dense.py build             -> work/train_dense_feat.parquet   (S1-dropout, records kept as distractors)
  python dense.py universe          -> work/train_univ_feat.parquet    (smaller universe: 50% of entities
                                        removed together with their records, unowned records kept)
"""
import hashlib
import sys
import time

import numpy as np
import pandas as pd

import config
import features
import ranker1

DROP_RATE = 0.186  # 4.68 / (1 - 0.186) ~= 5.75 records per S1


def dropped_s1(df, rate=DROP_RATE, salt="drop"):
    """Deterministic S1 dropout independent of the fold hash."""
    is1 = (df.src == 1).values
    ids = df.entity_id.values
    h = np.array([int(hashlib.md5((x + "|" + salt).encode()).hexdigest(), 16) % 10000 for x in ids[is1]])
    drop = np.zeros(len(df), bool)
    drop[np.flatnonzero(is1)[h < rate * 10000]] = True
    return drop


UNIV_RATE = 0.5


def universe_removed(df):
    """Records absent from the simulated smaller universe: dropped S1s and their true S2/S3 records."""
    from data_loader import read_ground_truth
    gone = dropped_s1(df, UNIV_RATE, "univ")
    gt = read_ground_truth()
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owned_by_gone = [o for s1 in df.entity_id.values[gone] for o in gt[s1]]
    gone[pos.reindex(owned_by_gone).values] = True
    return gone


def build(n_keep=25, mode="dense"):
    t0 = time.time()
    need = sorted(set(features.STR_COLS) | {"entity_id", "src", "country", "n_is_domain", "n_has_dba",
                                            "n_has_phone", "n_indic", "a_empty", "a_ncomp"})
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=need)
    drop = dropped_s1(df) if mode == "dense" else universe_removed(df)
    # ambiguity statistics are recomputed as if the dropped records did not exist
    df_vis = df[~drop]
    df = features.add_frequency_columns(df)
    freq = features.add_frequency_columns(df_vis.copy())[features.FREQ_COLS]
    for c in features.FREQ_COLS:
        df.loc[~drop, c] = freq[c].values
    print(f"dropped {drop.sum():,} of {(df.src == 1).sum():,} S1", flush=True)
    pairs = pd.read_parquet(config.work("train_cands.parquet"))
    pairs = pairs[~drop[pairs.q.values] & ~drop[pairs.p.values]].reset_index(drop=True)
    print(f"dense candidate pairs {len(pairs):,} ({time.time() - t0:.0f}s)", flush=True)
    seg = df.country.values[pairs.q.values]
    models, full = ranker1.train_models(pairs, seg)
    full.save_model(config.work(f"stage1_model_{mode}.txt"))
    score = ranker1.oof_scores(pairs, seg, models)
    r = features.group_rank(pairs.q.values, score)
    print(f"  dense stage-1 top{n_keep}: positives kept {pairs.y.values[r <= n_keep].sum():,} / {pairs.y.sum():,}")
    pairs = ranker1.top_n(pairs, score, n_keep)
    print(f"pruned -> {len(pairs):,} ({time.time() - t0:.0f}s)", flush=True)
    # reuse pairwise features from the normal build where the pair exists
    base_cols = features.PAIR_FEATS
    base = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p"] + base_cols)
    m = pairs[["q", "p"]].merge(base, on=["q", "p"], how="left")
    missing = m[base_cols[0]].isna().values
    print(f"pairs needing new pairwise features: {missing.sum():,}", flush=True)
    del base
    if missing.any():
        Xn = features.pair_features(df, pairs[missing].reset_index(drop=True))
        m.loc[missing, base_cols] = Xn[base_cols].values
    X = m[base_cols].reset_index(drop=True)
    # record-level flags and (dense) frequency columns
    for c in ("n_is_domain", "n_has_dba", "n_has_phone", "n_indic", "a_empty", "a_ncomp"):
        X["p_" + c] = df[c].values[pairs.p.values]
    X["q_a_empty"] = df["a_empty"].values[pairs.q.values]
    X["q_a_ncomp"] = df["a_ncomp"].values[pairs.q.values]
    X["src3"] = (df["src"].values[pairs.p.values] == 3).astype(np.int8)
    for c in features.FREQ_COLS:
        X["q_" + c] = df[c].values[pairs.q.values]
        X["p_" + c] = df[c].values[pairs.p.values]
    C = features.context_features(pairs.drop(columns=["s1_score", "s1_rank"]))
    out = pd.concat([pairs, C.drop(columns=[c for c in C.columns if c in pairs.columns]), X], axis=1)
    # identical column order to the normal feature table
    ref = __import__("pyarrow.parquet", fromlist=["x"]).read_schema(config.work("train_feat.parquet")).names
    out = out[ref]
    out.to_parquet(config.work("train_dense_feat.parquet" if mode == "dense" else "train_univ_feat.parquet"), index=False)
    print(f"dense features {out.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "build"
    build(mode="dense" if arg == "build" else "universe")
