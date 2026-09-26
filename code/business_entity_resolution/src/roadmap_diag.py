"""Evidence for the improvement roadmap (final system v3 @ thr0.9; dev folds 1-4 only).

  python roadmap_diag.py blocking   taxonomy of true pairs lost before stage 2
  python roadmap_diag.py errors     FN / FP taxonomy of the final system
  python roadmap_diag.py numbers    house-number differences: true matches vs unowned near-copies
"""
import sys

import numpy as np
import pandas as pd

import config
from data_loader import read_ground_truth
from prep import fold_of

DF_COLS = ["entity_id", "src", "country", "n_core", "n_compact", "a_norm", "a_first_num", "a_nums",
           "a_empty", "n_indic", "n_is_domain"]


def _load():
    df = pd.read_parquet(config.work("train_v1.parquet"), columns=DF_COLS)
    gt = read_ground_truth()
    return df, gt


def _true_pairs(df, gt):
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    q = [s for s, v in gt.items() for _ in v if fold_of(s) != 0]
    p = [o for s, v in gt.items() for o in v if fold_of(s) != 0]
    return pd.DataFrame({"q": pos.reindex(q).values, "p": pos.reindex(p).values})


def _describe(df, T):
    a, b = df.iloc[T.q.values].reset_index(drop=True), df.iloc[T.p.values].reset_index(drop=True)
    at = [set(x.split()) for x in a.n_core]; bt = [set(x.split()) for x in b.n_core]
    jac = np.array([len(x & y) / max(len(x | y), 1) for x, y in zip(at, bt)])
    both_num = (a.a_first_num.values != "") & (b.a_first_num.values != "")
    return pd.DataFrame({
        "other addr empty": b.a_empty.values == 1,
        "name share no token (DBA/renamed)": jac == 0,
        "Indic-script other name": b.n_indic.values == 1,
        "domain-name other": b.n_is_domain.values == 1,
        "house no. differs (both present)": both_num & (a.a_first_num.values != b.a_first_num.values),
        "country": a.country.values})


def blocking():
    df, gt = _load()
    T = _true_pairs(df, gt)
    C = pd.read_parquet(config.work("train_cands.parquet"), columns=["q", "p"])
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p"])
    key = lambda d: d.q.values.astype(np.int64) * 20_000_000 + d.p.values
    in_c, in_f = np.isin(key(T), key(C)), np.isin(key(T), key(F))
    print(f"dev true pairs {len(T):,}: retrieved {in_c.mean():.4f}, kept after stage-1 top-25 {in_f.mean():.4f}")
    D = _describe(df, T)
    for name, m in (("never retrieved", ~in_c), ("retrieved but pruned", in_c & ~in_f), ("all true pairs", np.ones(len(T), bool))):
        sub = D[m]
        print(f"\n{name}: {m.sum():,}")
        print(sub.drop(columns="country").mean().round(3).to_string())
        print("   by country:", sub.country.value_counts(normalize=True).round(3).to_dict())
    miss = D[~in_c]
    cat = np.select([miss["other addr empty"], miss["name share no token (DBA/renamed)"], miss["Indic-script other name"]],
                    ["empty address", "no shared name token", "Indic script"], "other")
    print("\nexclusive categories of never-retrieved:", pd.Series(cat).value_counts(normalize=True).round(3).to_dict())


def errors():
    df, gt = _load()
    F = pd.read_parquet(config.work("train_feat.parquet"),
                        columns=["q", "p", "y", "fold", "n_tset", "a_tset", "p_a_empty", "a_first_num_eq", "a_nn_q", "a_nn_p",
                                 "p_n_indic", "n_skel_ratio", "n_ratio"])
    F["prob"] = np.load(config.work("oof_s3inv3.npy"))
    F = F[F.fold != 0]
    best = F.groupby("p").prob.transform("max")
    acc = (F.prob >= 0.9) & (F.prob >= best)
    fn = F[(F.y == 1) & ~acc]
    fp = F[(F.y == 0) & acc]
    both_num = (F.a_nn_q > 0) & (F.a_nn_p > 0)
    def cls(X):
        bn = (X.a_nn_q > 0) & (X.a_nn_p > 0)
        return np.select([X.p_a_empty == 1, X.n_tset < 50, X.p_n_indic == 1,
                          bn & (X.a_first_num_eq == 0) & (X.n_tset >= 90),
                          (X.n_skel_ratio >= 90) & (X.n_ratio < 90), X.a_tset < 70],
                         ["no address (name-only)", "name mostly different (DBA/renamed)", "Indic-script name",
                          "same name, house no. differs", "typo in name", "address differs substantially"], "other")
    for name, X in (("FALSE NEGATIVES (true, not accepted)", fn), ("FALSE POSITIVES (accepted, wrong)", fp)):
        c = pd.Series(cls(X), index=X.index)
        band = pd.cut(X.prob, [-0.01, 0.1, 0.5, 0.9, 1.0], labels=["<0.1", "0.1-0.5", "0.5-0.9", ">=0.9"])
        t = pd.crosstab(c, band, margins=True)
        print(f"\n===== {name}: {len(X):,}")
        print(t.to_string())


def numbers():
    df, gt = _load()
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "n_tset", "a_tset"])
    s = F[(F.n_tset >= 90) & (F.a_tset >= 85)].sort_values("n_tset", ascending=False).drop_duplicates("p")
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owned = np.zeros(len(df), bool)
    owned[pos.reindex([o for v in gt.values() for o in v]).values] = True
    a, b = df.a_first_num.values[s.q.values], df.a_first_num.values[s.p.values]
    m = (a != "") & (b != "") & (a != b)
    s = s[m]; a, b = a[m], b[m]
    kind = np.where(s.y.values == 1, "true match", np.where(~owned[s.p.values], "unowned distractor", "other S1's record"))
    from rapidfuzz.distance import Levenshtein
    ai, bi = np.array([int(x) for x in a]), np.array([int(x) for x in b])
    R = pd.DataFrame({"kind": kind, "absdiff": np.abs(ai - bi), "lev": [Levenshtein.distance(x, y) for x, y in zip(a, b)],
                      "same_len": [len(x) == len(y) for x, y in zip(a, b)],
                      "prefix_of": [x.startswith(y) or y.startswith(x) for x, y in zip(a, b)],
                      "other_nums_contain": [x in set(n.split()) for x, n in zip(a, df.a_nums.values[s.p.values])]})
    R["one_digit_edit"] = R.lev == 1
    R["small_offset<=12"] = R.absdiff <= 12
    R["same_parity"] = (ai % 2) == (bi % 2)
    print(R.groupby("kind").agg(n=("lev", "size"), one_digit_edit=("one_digit_edit", "mean"),
                                small_offset=("small_offset<=12", "mean"), same_len=("same_len", "mean"),
                                truncation=("prefix_of", "mean"), same_parity=("same_parity", "mean"),
                                s1_no_elsewhere=("other_nums_contain", "mean")).round(3).to_string())
    print("\nabsdiff quantiles:\n", R.groupby("kind").absdiff.quantile([0.25, 0.5, 0.75]).unstack().round(1).to_string())


if __name__ == "__main__":
    {"blocking": blocking, "errors": errors, "numbers": numbers}[sys.argv[1]]()
