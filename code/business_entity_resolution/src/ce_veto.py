"""Near-copy veto with the cross-encoder.

The CE never scored pairs with stage-2 probability >= 0.999 (on dev they held no missed matches and
2.5% of false merges).  Test, however, carries ~0.5 extra near-exact *unowned* copies per S1 (label-free
estimate, section 5.7), and near-exact copies sit exactly there.  This scores the ACCEPTED pairs with
p2 >= 0.999 with the CE (same half-model routing as ce_train: dev by fold, test by S1 hash, France by the
self-trained models) and drops those the CE rejects with logit < tau.

Gate (pre-registered, see apply): tau chosen on one random half of a 25% dev S1 sample, confirmed on the
other half (gain >= +0.00014 with bootstrap CI above 0); test data never influences the decision.

  python ce_veto.py export              (CPU box)  -> work/veto_dev*.parquet, work/veto_test.parquet
  python ce_veto.py score               (GPU)      -> work/veto_{dev,test}_ce.npy
  python ce_veto.py apply IN.tsv OUTDIR            dev grid report, chosen tau, filtered matching file
"""
import os
import sys

import numpy as np
import pandas as pd

import config

P2_VETO = 0.999
TAG = "s3inv3n_ce"
GRID = [-8, -7, -6, -5, -4, -3, -2, -1, 0]


def export():
    import decision
    from prep import fold_of
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src"])
    eid = df.entity_id.values
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    F["prob"] = np.load(config.work(f"oof_{TAG}.npy"))
    F["p2"] = np.load(config.work("goof_inv3n__normal.npy"))
    sel = decision.threshold_policy(F, 0.9)
    sel = sel[sel.fold.values != 0]
    dev_s1 = np.array([x for x in eid[(df.src == 1).values] if fold_of(x) != 0])
    samp = np.sort(np.random.default_rng(0).choice(dev_s1, len(dev_s1) // 4, replace=False))
    q_id = eid[sel.q.values]
    keep = pd.Series(q_id).isin(set(samp)).values
    s = sel[keep]
    pd.DataFrame({"q_id": q_id[keep], "p_id": eid[s.p.values], "y": s.y.values.astype(np.int8),
                  "fold": s.fold.values.astype(np.int8), "p2": s.p2.values.astype(np.float32),
                  "p3": s.prob.values.astype(np.float32)}).to_parquet(config.work("veto_dev.parquet"), index=False)
    pd.DataFrame({"entity_id": samp}).to_parquet(config.work("veto_dev_s1.parquet"), index=False)
    T = pd.read_parquet(config.work("test_feat.parquet"), columns=["q", "p"])
    T["prob"] = np.load(config.work(f"test_prob_{TAG}.npy"))
    T["p2"] = np.load(config.work("test_p2_s3inv3n.npy"))
    dt = pd.read_parquet(config.work("test_v1.parquet"), columns=["entity_id", "country"])
    st = decision.threshold_policy(T, 0.9)
    st = st[st.p2.values >= P2_VETO]
    pd.DataFrame({"q_id": dt.entity_id.values[st.q.values], "p_id": dt.entity_id.values[st.p.values],
                  "country": dt.country.values[st.q.values], "p2": st.p2.values.astype(np.float32),
                  "p3": st.prob.values.astype(np.float32)}).to_parquet(config.work("veto_test.parquet"), index=False)
    print(f"dev sample: {len(samp):,} S1, {keep.sum():,} accepted pairs ({(s.p2.values >= P2_VETO).sum():,} with p2 >= {P2_VETO}) "
          f"| test accepted pairs with p2 >= {P2_VETO}: {len(st):,}")


def score():
    import torch
    import ce_train as C
    tok = C.AutoTokenizer.from_pretrained(C.MODEL)

    def model(w):
        m = C.CrossEncoder().to(C.DEV)
        m.load_state_dict(torch.load(config.work(w), map_location=C.DEV))
        return m

    D = pd.read_parquet(config.work("veto_dev.parquet"))
    m = D.p2.values >= P2_VETO
    ids = pd.Index(pd.unique(np.r_[D.q_id.values[m], D.p_id.values[m]]))
    text = C.texts("train", ids)
    out = np.full(len(D), np.nan, np.float32)
    for h, folds in (("A", (3, 4)), ("B", (1, 2))):           # the half-model that did not train on the fold
        mm = m & np.isin(D.fold.values, folds)
        out[mm] = C.score(model(f"ce_model_{h}.pt"), tok, text, ids.get_indexer(D.q_id.values[mm]), ids.get_indexer(D.p_id.values[mm]))
    np.save(config.work("veto_dev_ce.npy"), out)
    print(f"dev scored {m.sum():,}", flush=True)
    T = pd.read_parquet(config.work("veto_test.parquet"))
    ids = pd.Index(pd.unique(np.r_[T.q_id.values, T.p_id.values]))
    text = C.texts("test", ids)
    half = pd.util.hash_array(T.q_id.values) % 2                 # same routing as ce_train.score_all
    fr = T.country.values == "France"
    out = np.empty(len(T), np.float32)
    for h in (0, 1):
        for is_fr in (False, True):
            mm = (half == h) & (fr == is_fr)
            w = f"ce_{'france_' if is_fr else ''}model_{'AB'[h]}.pt"
            out[mm] = C.score(model(w), tok, text, ids.get_indexer(T.q_id.values[mm]), ids.get_indexer(T.p_id.values[mm]))
    np.save(config.work("veto_test_ce.npy"), out)
    print(f"test scored {len(T):,}", flush=True)


def apply(src, out_dir):
    """Pre-registered gate (project rule: a change must GAIN on dev before it earns a submission):
    the dev S1 sample is split in two random halves by hash.  tau is chosen on the selection half as the
    grid value with the largest macro-F0.5 gain (none > 0 -> no veto); on the untouched confirmation half
    the gain must be >= +0.00014 (2x seed std) with a paired-bootstrap 95% CI above 0.  Only then is the
    filtered test file written.  Test veto rates are printed as a diagnostic and never used to decide."""
    from data_loader import read_ground_truth
    from evaluation import evaluate_predictions, paired_bootstrap
    from submission import write_sets
    D = pd.read_parquet(config.work("veto_dev.parquet"))
    ce = np.load(config.work("veto_dev_ce.npy"))
    s1 = pd.read_parquet(config.work("veto_dev_s1.parquet")).entity_id.values
    gt = read_ground_truth()
    cand = D.p2.values >= P2_VETO
    half = pd.util.hash_array(s1.astype(object), hash_key="veto_split_00001") % 2
    halves = {"selection": s1[half == 0], "confirmation": s1[half == 1]}

    def f05(ids, keep):
        pred = {i: set() for i in ids}
        for a, b in zip(D.q_id.values[keep], D.p_id.values[keep]):
            if a in pred:
                pred[a].add(b)
        return evaluate_predictions(pred, gt, ids=ids, verbose=False, return_scores=True)

    T = pd.read_parquet(config.work("veto_test.parquet"))
    tpath = config.work("veto_test_ce.npy")
    tce = np.load(tpath) if os.path.exists(tpath) else np.full(len(T), np.nan, np.float32)   # gate needs dev only
    print(f"dev sample {len(s1):,} S1 | accepted {len(D):,}, with p2 >= {P2_VETO}: {cand.sum():,} "
          f"(false {(cand & (D.y.values == 0)).sum():,}) | test accepted with p2 >= {P2_VETO}: {len(T):,}")
    sel_ids = halves["selection"]
    base = f05(sel_ids, np.ones(len(D), bool))
    best, best_d = None, 0.0
    for t in GRID:
        veto = cand & (ce < t)
        d = float(np.mean(f05(sel_ids, ~veto)["_scores"]) - np.mean(base["_scores"]))
        print(f"  [selection] tau {t:+d}: vetoes {veto.sum():,} ({(veto & (D.y.values == 0)).sum():,} false / "
              f"{(veto & (D.y.values == 1)).sum():,} true, all dev sample) dF0.5 {d:+.6f} | "
              f"diagnostic: dev veto rate {veto.sum() / max(cand.sum(), 1):.5f} vs test {np.mean(tce < t):.5f}", flush=True)
        if d > best_d:
            best, best_d = t, d
    if best is None:
        print("GATE FAIL: no tau gains on the selection half -> no veto, no submission")
        return
    conf = halves["confirmation"]
    veto = cand & (ce < best)
    r0, r1 = f05(conf, np.ones(len(D), bool)), f05(conf, ~veto)
    d, lo, hi = paired_bootstrap(r0["_scores"], r1["_scores"])
    ok = d >= 0.00014 and lo > 0
    print(f"[confirmation] tau {best:+d}: dF0.5 {d:+.6f} [{lo:+.6f}, {hi:+.6f}] -> GATE {'PASS' if ok else 'FAIL'} "
          f"(needs >= +0.00014 and CI above 0)")
    if not ok:
        return
    if not os.path.exists(tpath):
        print("gate passed; test scores not ready yet -> rerun apply to write the file")
        return
    drop = set(zip(T.q_id.values[tce < best], T.p_id.values[tce < best]))
    M = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
    s1t = M.iloc[:, 0].values
    pred = {a: set(x for x in b.split(",") if x) for a, b in zip(s1t, M.iloc[:, 1].values)}
    n0 = sum(len(v) for v in pred.values())
    for q, p in drop:
        if q in pred:
            pred[q].discard(p)
    os.makedirs(out_dir, exist_ok=True)
    write_sets(os.path.join(out_dir, "matching_results.tsv"), s1t, pred, "matched_entity_ids")
    print(f"tau {best:+d}: removed {n0 - sum(len(v) for v in pred.values()):,} links from {src} -> {out_dir}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"export": export, "score": score, "apply": lambda: apply(sys.argv[2], sys.argv[3])}[cmd]()
