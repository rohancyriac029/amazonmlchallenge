"""E1 shift suite: inject realistic near-copy distractors into a dev fold.

Step 1 (measure): characterise how *real* unowned near-copy records in train differ
from the S1 they resemble (training labels only).  Step 2 (build): for S1s of the
evaluation fold, add perturbed copies of their own true records following that
empirical distribution, at the rate observed label-free on test (+0.6 near-copies
per S1).  Parameters are fixed before any F0.5 is computed.

  python sim_shift.py measure
  python sim_shift.py build      -> work/train_sim_v1.parquet (train records + injected distractors)
"""
import re
import sys

import numpy as np
import pandas as pd

import config
from data_loader import read_ground_truth

STRONG = dict(n_tset=90, a_tset=85)


def measure():
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "n_tset", "a_tset"])
    df = pd.read_parquet(config.work("train_v1.parquet"),
                         columns=["entity_id", "n_core", "n_legal", "a_norm", "a_first_num", "a_words", "a_empty"])
    gt = read_ground_truth()
    owned = np.zeros(len(df), bool)
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owned[pos.reindex([o for v in gt.values() for o in v]).values] = True
    s = F[(F.n_tset >= STRONG["n_tset"]) & (F.a_tset >= STRONG["a_tset"])]
    # best S1 per strong near-copy record
    s = s.sort_values("n_tset", ascending=False).drop_duplicates("p")
    s = s.assign(unowned=~owned[s.p.values])
    rows = []
    for kind, g in (("unowned near-copy", s[s.unowned]), ("true match (strong)", s[s.y == 1])):
        g = g.sample(min(300_000, len(g)), random_state=0)
        a, b = df.iloc[g.q.values].reset_index(drop=True), df.iloc[g.p.values].reset_index(drop=True)
        name_same = (a.n_core.values == b.n_core.values)
        num_both = (a.a_first_num.values != "") & (b.a_first_num.values != "")
        num_diff = num_both & (a.a_first_num.values != b.a_first_num.values)
        words_same = np.array([set(x.split()) == set(y.split()) for x, y in zip(a.a_words, b.a_words)])
        extra_tok = np.array([len(set(y.split()) - set(x.split())) > 0 for x, y in zip(a.n_core, b.n_core)])
        miss_tok = np.array([len(set(x.split()) - set(y.split())) > 0 for x, y in zip(a.n_core, b.n_core)])
        rows.append({"kind": kind, "n": len(g), "name identical": name_same.mean(),
                     "name has extra token": extra_tok.mean(), "name lacks a token": miss_tok.mean(),
                     "house no. differs (both present)": num_diff.sum() / max(num_both.sum(), 1),
                     "addr words identical": words_same.mean(),
                     "fully identical (name+no.+words)": (name_same & ~num_diff & words_same).mean()})
    print(pd.DataFrame(rows).set_index("kind").T.round(3).to_string())
    print(f"\nunowned strong near-copies: {int(s.unowned.sum()):,}; per S1 (train) "
          f"{s.unowned.sum() / (df.entity_id.str.startswith('S1').sum()):.3f}")


EVAL_FOLD = 4          # dev fold receiving injected distractors (holdout fold 0 is never touched)
RATE = 0.6             # extra near-copies per S1: label-free test statistic (test 3.71 vs train 3.13 US)
P_NUM, P_ADD, P_DROP = 0.90, 0.47, 0.11   # measured on real unowned near-copies (measure())
_NUM_RE = re.compile(r"\d+")


def extra_token_vocab(df, F, owned, n=300_000):
    """Tokens that real unowned near-copies add to the S1 name (empirical distribution)."""
    s = F[(F.n_tset >= STRONG["n_tset"]) & (F.a_tset >= STRONG["a_tset"])]
    s = s[~owned[s.p.values]].drop_duplicates("p").sample(n, random_state=1)
    toks = []
    for a, b in zip(df.n_cons.values[s.q.values], df.n_cons.values[s.p.values]):
        toks += list(set(b.split()) - set(a.split()))
    vc = pd.Series(toks).value_counts()
    return vc.index.values, (vc.values / vc.values.sum())


def perturb(name, addr, rng, vocab, probs):
    if addr and rng.random() < P_NUM:
        m = _NUM_RE.search(addr)
        if m:
            old = int(m.group())
            delta = int(rng.choice([-1, 1]) * rng.integers(1, 13))
            new = max(1, old + delta) if old > 1 else old + abs(delta)
            addr = addr[:m.start()] + str(new) + addr[m.end():]
    toks = name.split()
    if rng.random() < P_ADD:
        t = str(rng.choice(vocab, p=probs)).title()
        toks = toks + [t] if rng.random() < 0.7 else [t] + toks
    if len(toks) > 2 and rng.random() < P_DROP:
        toks.pop(int(rng.integers(0, len(toks))))
    return " ".join(toks), addr


def build():
    from data_loader import normalize_frame
    import translit_dict
    from prep import fold_of
    df = pd.read_parquet(config.work("train_v1.parquet"))
    F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "n_tset", "a_tset"])
    gt = read_ground_truth()
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    owned = np.zeros(len(df), bool)
    owned[pos.reindex([o for v in gt.values() for o in v]).values] = True
    vocab, probs = extra_token_vocab(df, F, owned)
    print("top injected tokens:", list(vocab[:15]))
    del F
    rng = np.random.default_rng(2026)
    s1 = df[(df.src == 1).values]
    s1 = s1[[fold_of(x) == EVAL_FOLD for x in s1.entity_id.values]]
    rows = []
    for sid, sname, saddr, cty in zip(s1.entity_id.values, s1.business_name.values, s1.business_address.values,
                                      s1.country.values):
        k = rng.poisson(RATE)
        if k == 0:
            continue
        base = sorted(gt[sid])
        for j in range(k):
            if base:
                b = pos[base[int(rng.integers(0, len(base)))]]
                name, addr, src = df.business_name.values[b], df.business_address.values[b], df.src.values[b]
            else:
                name, addr, src = sname, saddr, np.int8(rng.integers(2, 4))
            n2, a2 = perturb(name, addr, rng, vocab, probs)
            rows.append((f"S{src}-SIM{len(rows):07d}", n2, a2, cty, np.int8(src)))
    inj = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country", "src"])
    print(f"injected {len(inj):,} distractors for {len(s1):,} fold-{EVAL_FOLD} S1 ({len(inj) / len(s1):.3f}/S1)")
    inj = normalize_frame(inj, translit_dict.load(config.work("translit_dict.json")))
    out = pd.concat([df, inj[df.columns]], ignore_index=True)
    out.to_parquet(config.work("train_sim_v1.parquet"), index=False)
    print("examples:\n", inj[["business_name", "business_address"]].head(8).to_string())


if __name__ == "__main__":
    {"measure": measure, "build": build}[sys.argv[1]]()
