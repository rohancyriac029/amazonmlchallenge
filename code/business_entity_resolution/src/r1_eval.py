"""R1 (evidence-conditional decision) and R2/X2 (stage-3 under injection) evaluation.

  1. precondition (label-free): distractor share among near-copies whose house numbers AGREE,
     train vs test, from the extra-name-token signature (house-number signature is undefined there)
  2. dev folds 1-4: thr0.9 vs expF vs conditional, bootstrap + per-class
  3. shift suite fold 4 (clean / injected): same policies, stage 2 alone vs stage 2+3 (X2)
Nothing here touches the holdout fold.
"""
import numpy as np
import pandas as pd

import config
import decision
import train
from data_loader import read_ground_truth
from e_compare import classify, CLS_COLS
from evaluation import evaluate_predictions, paired_bootstrap
from prep import fold_of

STRONG = dict(n_tset=90, a_tset=85)


def agree_stratum_pi():
    gt = read_ground_truth()
    out = {}
    for split in ("train", "test"):
        F = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=["q", "p", "n_tset", "a_tset"])
        Ns = pd.read_parquet(config.work(f"{split}_feat_N.parquet"), columns=["num_state"]).num_state.values
        df = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["entity_id", "n_core", "country"])
        m = (F.n_tset.values >= STRONG["n_tset"]) & (F.a_tset.values >= STRONG["a_tset"]) & (Ns == 1)
        s = F[m].sort_values("n_tset", ascending=False).drop_duplicates("p")
        extra = np.array([len(set(y.split()) - set(x.split())) > 0
                          for x, y in zip(df.n_core.values[s.q.values], df.n_core.values[s.p.values])])
        out[split] = (s, extra, df)
    s, extra, df = out["train"]
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owned = np.zeros(len(df), bool)
    owned[pos.reindex([o for v in gt.values() for o in v]).values] = True
    un = ~owned[s.p.values]
    r_d, r_t = extra[un].mean(), extra[~un].mean()
    est = lambda e: (e.mean() - r_t) / (r_d - r_t)
    print(f"[precondition] number-agree near-copies: extra-token rate distractor {r_d:.3f} vs true {r_t:.3f}")
    print(f"  train: actual distractor share {un.mean():.3f}, signature estimate {est(extra):.3f}")
    s_t, e_t, df_t = out["test"]
    cty = df_t.country.values[s_t.q.values]
    print(f"  test : signature estimate {est(e_t):.3f}  by country " +
          str({c: round(est(e_t[cty == c]), 3) for c in np.unique(cty)}))


def policies(F):
    return {"thr0.9": lambda: decision.threshold_policy(F, 0.9),
            "expF_gate0.6": lambda: decision.gated_expected_f_policy(F, gate=0.6, miss_mass=0.3),
            "cond_num": lambda: decision.conditional_policy(F)}


def dev_eval():
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src"])
    s1 = df.entity_id.values[(df.src == 1).values]
    ids = np.sort([x for x in s1 if fold_of(x) != 0])
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=CLS_COLS)
    F["num_agree"] = pd.read_parquet(config.work("train_feat_N.parquet"), columns=["num_state"]).num_state.values == 1
    F["prob"] = np.load(config.work("oof_s3inv3n.npy"))
    cls = pd.Series(classify(F), index=F.index)
    dev = F.fold.values != 0
    res, accs = {}, {}
    for name, fn in policies(F).items():
        sel = fn()
        r = evaluate_predictions(decision.to_sets(df, sel, ids), gt, ids=ids, verbose=False, return_scores=True)
        res[name] = r
        a = np.zeros(len(F), bool); a[sel.index.values] = True; accs[name] = a
        print(f"[dev] {name:13s} F0.5={r['macro_f05']:.5f} P={r['macro_precision']:.4f} R={r['macro_recall']:.4f} "
              f"single={r['singleton_acc']:.4f} FP={r['false_merges']:,} missed={r['missed']:,}")
    d, lo, hi = paired_bootstrap(res["thr0.9"]["_scores"], res["cond_num"]["_scores"])
    print(f"[dev] cond_num - thr0.9: {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    train.log_experiment({"tag": "R1_cond_num_dev", "delta_vs_thr0.9": d, "ci": [lo, hi],
                          **{f"f_{k}": v["macro_f05"] for k, v in res.items()}})
    y = F.y.values == 1
    rows = []
    for c in sorted(cls.unique()):
        m = dev & (cls.values == c)
        rows.append({"class": c, "FN thr0.9": int((m & y & ~accs["thr0.9"]).sum()), "FN cond": int((m & y & ~accs["cond_num"]).sum()),
                     "FP thr0.9": int((m & ~y & accs["thr0.9"]).sum()), "FP cond": int((m & ~y & accs["cond_num"]).sum())})
    print(pd.DataFrame(rows).set_index("class").to_string())


def suite_eval():
    gt = read_ground_truth()
    dtr = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src"])
    ds = pd.read_parquet(config.work("train_sim_v1.parquet"), columns=["entity_id", "src"])
    s1 = dtr.entity_id.values[(dtr.src == 1).values]
    ids = np.sort([x for x in s1 if fold_of(x) == 4])
    sets = {
        "clean": (pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p"]),
                  pd.read_parquet(config.work("train_feat_N.parquet"), columns=["num_state"]).num_state.values, dtr,
                  np.load(config.work("goof_inv3n__normal.npy")), np.load(config.work("oof_s3inv3n.npy"))),
        "injected": (pd.read_parquet(config.work("train_sim_feat.parquet"), columns=["q", "p"]),
                     pd.read_parquet(config.work("train_sim_feat_N.parquet"), columns=["num_state"]).num_state.values, ds,
                     np.load(config.work("sim_p2_v3n.npy")), np.load(config.work("sim_prob_v3n.npy"))),
    }
    for setname, (F, ns, d, p2, p3) in sets.items():
        for stage, pr in (("stage2", p2), ("stage3", p3)):
            X = F.copy(); X["prob"] = pr; X["num_agree"] = ns == 1
            for name, fn in policies(X).items():
                r = evaluate_predictions(decision.to_sets(d, fn(), ids), gt, ids=ids, verbose=False)
                print(f"[suite fold4 {setname:8s}] {stage} {name:13s} F0.5={r['macro_f05']:.5f} FP={r['false_merges']:,} "
                      f"missed={r['missed']:,} single={r['singleton_acc']:.4f}")
                train.log_experiment({"tag": f"R1X2_{setname}_{stage}_{name}", "eval": f"fold4_{setname}",
                                      **{k: v for k, v in r.items() if not k.startswith("_")}})


if __name__ == "__main__":
    import sys
    {"pi": agree_stratum_pi, "dev": dev_eval, "suite": suite_eval}[sys.argv[1]]()
