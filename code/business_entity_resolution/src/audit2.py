"""Second-audit diagnostics (dev folds / unlabeled test only; holdout untouched).

  python audit2.py pi      X1: label-free distractor share among near-copies (train calibration + test estimate)
  python audit2.py loss    X8: error decomposition of the final system (v3@thr0.9) vs v3 expF on dev
  python audit2.py empty   X6: predicted P(no match) = prod(1-p) vs empirical, dev
  python audit2.py coh     X7: probability coherence across S1s claiming the same record, dev
"""
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


def _signatures(split):
    """For each pool record's best strong near-copy S1: house-number-differs and extra-name-token flags."""
    F = pd.read_parquet(config.work(f"{split}_feat.parquet"), columns=["q", "p", "n_tset", "a_tset"])
    df = pd.read_parquet(config.work(f"{split}_v1.parquet"), columns=["n_core", "a_first_num", "country"])
    s = F[(F.n_tset >= STRONG["n_tset"]) & (F.a_tset >= STRONG["a_tset"])]
    s = s.sort_values("n_tset", ascending=False).drop_duplicates("p")
    a_num, b_num = df.a_first_num.values[s.q.values], df.a_first_num.values[s.p.values]
    both = (a_num != "") & (b_num != "")
    diff = both & (a_num != b_num)
    a_core, b_core = df.n_core.values[s.q.values], df.n_core.values[s.p.values]
    extra = np.array([len(set(y.split()) - set(x.split())) > 0 for x, y in zip(a_core, b_core)])
    return s, both, diff, extra, df.country.values[s.q.values]


def pi():
    # signature rates for true matches (r1) and unowned distractors (r0), measured on train
    gt = read_ground_truth()
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id"])
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owned = np.zeros(len(df), bool)
    owned[pos.reindex([o for v in gt.values() for o in v]).values] = True
    s, both, diff, extra, cty = _signatures("train")
    un = ~owned[s.p.values]
    y = pd.read_parquet(config.work("train_feat.parquet"), columns=["y"]).y.values[s.index.values] == 1
    r_num_d, r_num_t = diff[un & both].mean(), diff[y & both].mean()
    r_tok_d, r_tok_t = extra[un].mean(), extra[y].mean()
    print(f"train signatures: house-no differs  distractor {r_num_d:.3f} vs true {r_num_t:.3f} | "
          f"extra token  distractor {r_tok_d:.3f} vs true {r_tok_t:.3f}")
    print(f"train actual distractor share among strong near-copy records: {un.mean():.3f}")
    n_s1_tr = int(df.entity_id.str.startswith("S1").sum())
    for split in ("train", "test"):
        s2, both2, diff2, extra2, cty2 = (s, both, diff, extra, cty) if split == "train" else _signatures("test")
        n_s1 = n_s1_tr if split == "train" else int(pd.read_parquet(config.work("test_v1.parquet"), columns=["src"]).src.eq(1).sum())
        for c in ["ALL"] + sorted(np.unique(cty2)):
            m = np.ones(len(s2), bool) if c == "ALL" else (cty2 == c)
            sh_num = diff2[m & both2].mean(); sh_tok = extra2[m].mean()
            pi_num = (sh_num - r_num_t) / (r_num_d - r_num_t)
            pi_tok = (sh_tok - r_tok_t) / (r_tok_d - r_tok_t)
            n_rec = m.sum()
            print(f"{split:5s} {c:7s} near-copy records {n_rec:>9,}  pi(house-no)={pi_num:.3f}  pi(extra-token)={pi_tok:.3f}"
                  + (f"  distractors/S1~{pi_num * n_rec / n_s1:.3f}" if c == 'ALL' else ""))


def _dev_frame(tag):
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
    F["prob"] = np.load(config.work(f"oof_{tag}.npy"))
    return F


def loss():
    import subprocess
    for pol in ("thr0.9", "expF_gate0.6"):
        print(f"\n===== v3 {pol}")
        subprocess.run([sys.executable, "analyze_loss.py", "s3inv3", pol], check=True)


def empty():
    F = _dev_frame("s3inv3")
    F = F[F.fold != 0]
    g = F.groupby("q")
    p_empty = g.prob.apply(lambda s: float(np.exp(np.log1p(-np.clip(s.values, 0, 1 - 1e-9)).sum())))
    has = g.y.max()
    d = pd.DataFrame({"pred": p_empty, "emp": (has == 0).astype(float)})
    d["bin"] = pd.cut(d.pred, [0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0], include_lowest=True)
    print(d.groupby("bin", observed=True).agg(n=("emp", "size"), predicted=("pred", "mean"), empirical=("emp", "mean")).round(4).to_string())
    print(f"overall predicted empty {d.pred.mean():.4f} vs empirical {d.emp.mean():.4f}")


def coh():
    F = _dev_frame("s3inv3")
    s = F.groupby("p").prob.agg(["sum", "max", "size"])
    multi = s[s["size"] > 1]
    print(f"records with >1 claimant: {len(multi):,}; share with sum P > 1: {(multi['sum'] > 1).mean():.4f}; "
          f"sum P > 1.5: {(multi['sum'] > 1.5).mean():.5f}")
    acc = F[F.prob >= 0.9]
    top2 = F.sort_values("prob", ascending=False).groupby("p").prob.nth(1)
    rival = acc.p.map(top2).fillna(0)
    print(f"accepted pairs (P>=0.9): {len(acc):,}; with a rival claimant P>0.5: {(rival > 0.5).mean():.4f}; "
          f"rival >= 0.9: {(rival >= 0.9).mean():.5f}")


if __name__ == "__main__":
    {"pi": pi, "loss": loss, "empty": empty, "coh": coh}[sys.argv[1]]()
