"""Roadmap evidence for the FINAL model (v3 + house-number features, thr 0.9); dev folds 1-4 only.

  python roadmap_diag2.py errors   residual FN/FP taxonomy incl. split of the former "other" class,
                                   residual house-number, name-only and DBA analyses
"""
import numpy as np
import pandas as pd

import config

TAG, S2 = "s3inv3n", "inv3n"
FEAT = ["q", "p", "y", "fold", "n_tset", "n_ratio", "n_skel_ratio", "a_tset", "a_ratio", "a_first_num_eq",
        "a_nn_q", "a_nn_p", "p_a_empty", "p_n_indic", "p_n_is_domain", "q_f_core_s1", "p_f_core_pool",
        "src3", "n_legal_jacc", "a_wcontain", "a_qnum_in_p", "s_conj_rank_q"]


def main():
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=FEAT)
    N = pd.read_parquet(config.work("train_feat_N.parquet"), columns=["num_logdiff", "num_neighbour", "num_same_parity"])
    S = pd.read_parquet(config.work(f"train_stack_{S2}.parquet"),
                        columns=["sib_name_max", "sib_addr_max", "sib_same_num", "n_sibs", "other_best_p"])
    F = pd.concat([F, N, S], axis=1)
    F["prob"] = np.load(config.work(f"oof_{TAG}.npy"))
    F = F[F.fold != 0]
    best = F.groupby("p").prob.transform("max")
    F["acc"] = (F.prob >= 0.9) & (F.prob >= best)
    bn = (F.a_nn_q > 0) & (F.a_nn_p > 0)
    F["cls"] = np.select(
        [F.p_a_empty == 1, F.n_tset < 50, F.p_n_indic == 1, bn & (F.a_first_num_eq == 0) & (F.n_tset >= 90),
         (F.n_skel_ratio >= 90) & (F.n_ratio < 90), F.a_tset < 70,
         # finer split of the former "other"
         (F.a_tset >= 95) & (F.n_tset < 90),
         (F.n_tset >= 90) & (F.a_tset < 85),
         (F.n_tset >= 90) & (F.a_tset >= 85) & ((F.a_nn_q == 0) | (F.a_nn_p == 0)),
         (F.n_tset >= 90) & (F.a_tset >= 85)],
        ["no address", "name different (DBA)", "Indic name", "same name, house no. differs", "typo in name",
         "address differs", "OTHER: same address, name partly differs", "OTHER: same name, locality/street partly differs",
         "OTHER: name+addr strong, one side has no number", "OTHER: name+addr strong, numbers agree"],
        "OTHER: partial name + partial address")
    fn = F[(F.y == 1) & ~F.acc]
    fp = F[(F.y == 0) & F.acc]
    print(f"dev FN (rejected candidates) {len(fn):,} | FP {len(fp):,}")
    band = lambda X: pd.cut(X.prob, [-0.01, 0.1, 0.5, 0.9, 1.0], labels=["<0.1", "0.1-0.5", "0.5-0.9", ">=0.9"])
    for name, X in (("FALSE NEGATIVES", fn), ("FALSE POSITIVES", fp)):
        t = pd.crosstab(X.cls, band(X), margins=True)
        t["share"] = (t["All"] / len(X)).round(3)
        print(f"\n===== {name}\n{t.to_string()}")
    # characteristics of FP / FN per class
    def prof(X, label):
        g = X.groupby("cls").agg(n=("y", "size"),
                                 s1_name_unique=("q_f_core_s1", lambda s: float((s <= 1).mean())),
                                 src3=("src3", "mean"),
                                 no_sibling=("n_sibs", lambda s: float((s == 0).mean())),
                                 sib_same_num=("sib_same_num", lambda s: float((s == 1).mean())),
                                 rival_gt_0_5=("other_best_p", lambda s: float((s > 0.5).mean())),
                                 retrieval_rank1=("s_conj_rank_q", lambda s: float((s == 1).mean())))
        print(f"\n--- {label} profile by class\n{g.round(3).to_string()}")
    prof(fn, "FN")
    prof(fp, "FP")
    # residual house-number class
    for name, X in (("FN", fn), ("FP", fp)):
        h = X[X.cls == "same name, house no. differs"]
        small = h.num_logdiff <= np.log1p(12)
        print(f"\nresidual house-no {name}: {len(h):,} | |delta|<=12: {small.mean():.3f} | neighbour flag: {(h.num_neighbour == 1).mean():.3f} | "
              f"same parity: {(h.num_same_parity == 1).mean():.3f} | S1 number elsewhere in record: {(h.a_qnum_in_p == 1).mean():.3f}")
    # name-only: potentially identifiable subset
    no = fn[fn.cls == "no address"]
    print(f"\nname-only FN {len(no):,}: S1 name unique among S1 {float((no.q_f_core_s1 <= 1).mean()):.3f} | "
          f"exact-name confident sibling (sib_name_max==100) {float((no.sib_name_max >= 100).mean()):.3f} | "
          f"both {float(((no.q_f_core_s1 <= 1) & (no.sib_name_max >= 100)).mean()):.3f} | prob 0.5-0.9 {float(((no.prob >= 0.5)).mean()):.3f}")
    nofp = fp[fp.cls == "no address"]
    print(f"name-only FP {len(nofp):,}: S1 name unique {float((nofp.q_f_core_s1 <= 1).mean()):.3f} | exact-name sibling {float((nofp.sib_name_max >= 100).mean()):.3f}")
    # DBA
    d = fn[fn.cls == "name different (DBA)"]
    exact = (d.a_first_num_eq == 1) & (d.a_tset >= 90)
    print(f"\nDBA FN {len(d):,}: exact number + address tset>=90: {exact.mean():.3f} | + confident sibling at same number: "
          f"{float((exact & (d.sib_same_num == 1)).mean()):.3f}")
    dfp = fp[fp.cls == "name different (DBA)"]
    print(f"DBA FP {len(dfp):,}: exact number + address tset>=90: {float(((dfp.a_first_num_eq == 1) & (dfp.a_tset >= 90)).mean()):.3f}")


if __name__ == "__main__":
    main()
