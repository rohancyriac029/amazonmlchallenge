"""Cross-encoder (CE) stage, step 2 (GPU): fine-tune a small multilingual cross-encoder on the
supplied training pairs and score the pairs chosen by ce_export.py.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters), used only as the
initialisation; it reads Devanagari and Latin script natively. No external data is used.

Two-way cross-fitting keeps the CE score out-of-fold for stage 3:
  model A: trained on dev folds 1-2, scores folds 3-4
  model B: trained on dev folds 3-4, scores folds 1-2
  holdout fold 0 and test: A or B, chosen by a hash of the S1 id (neither model saw them)
Every pair is scored by exactly ONE half-model, as the dev pairs stage 3 is trained on are.
(Averaging A and B for test was tried first: each half-model has its own saturation plateaus,
the mean lands between them, and dev-vs-test domain AUC rose to 0.89-0.95; see ce_domain.py.)

  python ce_train.py pilot [model ...] [--light]   200k-sample pilot(s); pre-registered gates on held-out S1s
  python ce_train.py train A|B      fine-tune one half                 -> work/ce{TAG}_model_{A,B}.pt
  python ce_train.py score [split]  -> work/ce{TAG}_logit_{train,test}.npy  (aligned with ce_pairs_*.parquet, NaN = not scored)
  python ce_train.py adapt France   self-training for a country absent from training (see adapt())
  python ce_train.py pack           -> work/ce_text_{train,test}.parquet: text of the entities in scored pairs, so the
                                       GPU step can run on another machine without the raw files

Env: ER_CE_MODEL (default intfloat/multilingual-e5-small; a local copy also works),
     ER_CE_TAG (output-name suffix for an additional cross-encoder, default ""),
     ER_CE_NTRAIN (pairs per half-model, default 1,000,000; 0 = every scored-region pair of its folds).
GPUs without bfloat16 (e.g. T4, P100) train in float16 with loss scaling.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import config

MODEL = os.environ.get("ER_CE_MODEL", "intfloat/multilingual-e5-small")
TAG = os.environ.get("ER_CE_TAG", "")
MAXLEN, LR, BATCH, SCORE_BATCH = 96, 5e-5, 128, 512
N_TRAIN = int(os.environ.get("ER_CE_NTRAIN", 1_000_000))   # pairs per half-model (one epoch)
ADAPT_LR, N_ADAPT, PSEUDO_HI, PSEUDO_LO = 2e-5, 300_000, 0.97, 0.03   # unseen-country self-training (ce_loco.py)
P2_MAX = 0.999                          # pairs above this stage-2 probability hold no dev FN and 2.5% of FP: not scored
HALVES = {"A": ((1, 2), (3, 4)), "B": ((3, 4), (1, 2))}   # name: (train folds, scored folds)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class CrossEncoder(torch.nn.Module):
    """Encoder over "S1 text [SEP] other text", mean-pooled, linear logit."""

    def __init__(self, model_name=MODEL, light=False):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        if light:   # fits a 6 GB GPU: frozen word embeddings + activation checkpointing (same maths otherwise)
            self.enc.embeddings.word_embeddings.weight.requires_grad_(False)
            self.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **_):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)


def load_pairs(split):
    """Pairs plus a compact text table: text[k] is "name | address" of the k-th entity used."""
    P = pd.read_parquet(config.work(f"ce_pairs_{split}.parquet"))
    P["k"] = np.arange(len(P))           # position in the exported file
    P = P[P.p2.values < P2_MAX].reset_index(drop=True)
    ids = pd.Index(pd.unique(np.concatenate([P.q_id.values, P.p_id.values])))
    text = texts(split, ids)
    P["qi"] = ids.get_indexer(P.q_id.values).astype(np.int32)
    P["h"] = (pd.util.hash_array(P.q_id.values) % 2).astype(np.int8)    # half-model for holdout/test pairs
    P["pi"] = ids.get_indexer(P.p_id.values).astype(np.int32)
    return P.drop(columns=["q_id", "p_id"]), text


def texts(split, ids):
    """ "name | address" per entity id, from the compact table written by pack() where it has them, else the raw TSVs."""
    tp = config.work(f"ce_text_{split}.parquet")
    text = (pd.read_parquet(tp).set_index("entity_id").text.reindex(ids).values if os.path.exists(tp)
            else np.full(len(ids), None, dtype=object))
    miss = np.flatnonzero(pd.isna(text))
    if len(miss):
        need = pd.Index(ids[miss])
        for k in (1, 2, 3):
            for ch in pd.read_csv(config.src_path(split, k), sep="\t", dtype=str, chunksize=1_000_000,
                                  usecols=["entity_id", "business_name", "business_address"]):
                pos = need.get_indexer(ch.entity_id.values)
                m = pos >= 0
                text[miss[pos[m]]] = (ch.business_name.fillna("").values[m] + " | " + ch.business_address.fillna("").values[m])
    assert all(isinstance(t, str) for t in text), "entity missing from the text source"
    return text


def pack():
    for split in ("train", "test"):
        P = pd.read_parquet(config.work(f"ce_pairs_{split}.parquet"), columns=["q_id", "p_id", "p2"])
        P = P[P.p2.values < P2_MAX]
        ids = pd.Index(pd.unique(np.concatenate([P.q_id.values, P.p_id.values])))
        pd.DataFrame({"entity_id": ids, "text": texts(split, ids)}).to_parquet(
            config.work(f"ce_text_{split}.parquet"), index=False, compression="zstd")
        print(f"{split}: {len(ids):,} entities packed", flush=True)


def batches(tok, text, qi, pi, size):
    """Tokenised batches, prepared one step ahead on a thread (fast tokenizers release the GIL)."""
    def enc(s):
        return tok(list(text[qi[s]]), list(text[pi[s]]), truncation="longest_first", max_length=MAXLEN,
                   padding=True, return_tensors="pt")
    slices = [slice(i, i + size) for i in range(0, len(qi), size)]
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(enc, s) for s in slices[:2]]
        for j in range(len(slices)):
            b = futs[j].result()
            futs[j] = None                       # release the batch, or every batch stays in RAM
            if j + 2 < len(slices):
                futs.append(ex.submit(enc, slices[j + 2]))
            yield slices[j], {k: v.to(DEV, non_blocking=True) for k, v in b.items()}


def sample_train(P, folds, n, seed):
    """Uncertain pairs (0.02 < p2 < 0.98) first, filled up with random decisive pairs (at least 30%)."""
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(P.fold.isin(folds).values)
    unc = cand[(P.p2.values[cand] > 0.02) & (P.p2.values[cand] < 0.98)]
    rest = np.setdiff1d(cand, unc)
    if n <= 0 or n >= len(cand):                 # every pair of these folds
        return rng.permutation(cand)
    n_unc = min(len(unc), int(n * 0.7))
    idx = np.concatenate([rng.choice(unc, n_unc, replace=False), rng.choice(rest, n - n_unc, replace=False)])
    return rng.permutation(idx)


def fit(P, text, idx, seed=0, model_name=MODEL, light=False, init=None, lr=LR):
    """Fine-tune on rows idx of P (columns qi, pi, y); init = state_dict path to continue from."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = CrossEncoder(model_name, light).to(DEV)
    if init:
        model.load_state_dict(torch.load(init, map_location=DEV))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = (len(idx) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.03 * steps)) * max(0.0, (steps - s) / steps))
    # length bucketing: sort within chunks of 50 batches, then shuffle the batch order
    ln = np.fromiter((len(text[a]) + len(text[b]) for a, b in zip(P.qi.values[idx], P.pi.values[idx])), np.int32, len(idx))
    ch = BATCH * 50
    idx = np.concatenate([c[np.argsort(l, kind="stable")] for c, l in
                          zip(np.array_split(idx, np.arange(ch, len(idx), ch)), np.array_split(ln, np.arange(ch, len(idx), ch)))])
    bl = np.array_split(idx, np.arange(BATCH, len(idx), BATCH))
    idx = np.concatenate([bl[j] for j in np.random.default_rng(seed).permutation(len(bl))])
    y = torch.tensor(P.y.values[idx], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)   # no-op under bfloat16
    model.train()
    t0, run = time.time(), 0.0
    for step, (s, b) in enumerate(batches(tok, text, P.qi.values[idx], P.pi.values[idx], BATCH)):
        with torch.autocast(DEV.type, dtype=AMP):
            loss = lossf(model(**b).float(), y[s].to(DEV))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step}/{steps} loss {run:.4f} ({(step + 1) * BATCH / (time.time() - t0):.0f} pairs/s)", flush=True)
    return model, tok


@torch.inference_mode()
def score(model, tok, text, qi, pi):
    """Logits in the original order; pairs are length-sorted so padding stays small."""
    model.eval()
    order = np.argsort(np.fromiter((len(text[a]) + len(text[b]) for a, b in zip(qi, pi)), np.int32, len(qi)), kind="stable")
    out = np.empty(len(qi), np.float32)
    t0 = time.time()
    for k, (s, b) in enumerate(batches(tok, text, qi[order], pi[order], SCORE_BATCH)):
        with torch.autocast(DEV.type, dtype=AMP):
            out[order[s]] = model(**b).float().cpu().numpy()
        if k % 2000 == 0:
            print(f"  scored {s.stop:,}/{len(qi):,} ({s.stop / (time.time() - t0):.0f} pairs/s)", flush=True)
    return out


def pilot(models=(MODEL,), light=False):
    """Pilot on a 200k-pair sample of folds 1-2, evaluated on held-out S1s of folds 3-4.

    Gates (fixed before running):
      one model  : stage-3 + CE combiner must cut pair errors at thr 0.9 by >= 2% vs stage 3 alone
      two models : the second (larger) model's combiner must cut errors by >= 10% vs the first's
    Every model gets the same sample, recipe and evaluation pairs; the ensemble of all is reported too."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    P, text = load_pairs("train")
    idx = sample_train(P, HALVES["A"][0], 200_000, 0)
    rng = np.random.default_rng(1)
    q = P.qi.values
    held = P.fold.isin(HALVES["A"][1]).values
    s1 = np.unique(q[held]); s1 = rng.choice(s1, len(s1) // 20, replace=False)
    ev = np.flatnonzero(held & np.isin(q, s1))
    fitq = np.isin(q[ev], s1[: len(s1) // 2])
    y, p3 = P.y.values[ev], P.p3.values[ev]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    yv = y[~fitq]
    err = lambda p: int(((yv == 1) & (p < 0.9)).sum() + ((yv == 0) & (p >= 0.9)).sum())
    ll = lambda p: float(-np.mean(yv * np.log(np.clip(p, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1))))
    ce = {}
    for name in models:
        t0 = time.time()
        model, tok = fit(P, text, idx, 0, name, light)
        ce[name] = score(model, tok, text, q[ev], P.pi.values[ev])
        print(f"[{os.path.basename(name)}] trained + scored in {time.time() - t0:.0f}s | CE alone AUC "
              f"{roc_auc_score(yv, ce[name][~fitq]):.4f} (stage 3 {roc_auc_score(yv, p3[~fitq]):.4f})", flush=True)
        del model
        torch.cuda.empty_cache()
    e3 = err(p3[~fitq])
    print(f"pilot eval: {len(yv):,} pairs of {len(s1) // 2:,} held-out S1s | stage 3 alone: errors@0.9 {e3:,}, logloss {ll(p3[~fitq]):.4f}")
    res = {}
    for label, names in [(os.path.basename(n), [n]) for n in models] + ([("ensemble", list(models))] if len(models) > 1 else []):
        X = np.c_[[lg(p3)] + [ce[n] for n in names]].T
        comb = LogisticRegression(C=1.0).fit(X[fitq], y[fitq]).predict_proba(X[~fitq])[:, 1]
        res[label] = err(comb)
        print(f"  stage 3 + {label:28s} errors@0.9 {res[label]:,} ({(res[label] - e3) / e3:+.1%}) | logloss {ll(comb):.4f}")
    first, last = os.path.basename(models[0]), os.path.basename(models[-1])
    if len(models) == 1:
        print("GATE", "PASS" if res[first] <= 0.98 * e3 else "FAIL")
    else:
        print(f"GATE {last} vs {first}: {(res[last] - res[first]) / res[first]:+.1%} ->",
              "PASS" if res[last] <= 0.90 * res[first] else "FAIL")


def train_half(name):
    P, text = load_pairs("train")
    model, _ = fit(P, text, sample_train(P, HALVES[name][0], N_TRAIN, {"A": 0, "B": 1}[name]))
    torch.save(model.state_dict(), config.work(f"ce{TAG}_model_{name}.pt"))


def adapt(country):
    """Self-training for a country absent from training, validated by the India-as-unseen simulation
    (ce_loco.py: +0.0121 India dev macro F0.5, also with the combiner frozen).  Each half-model continues
    fine-tuning on confident final-model predictions for that country's test pairs (pseudo-labels: stage-3
    probability >= PSEUDO_HI -> match, <= PSEUDO_LO -> non-match; no test label exists or is used), mixed
    with labelled pairs of its own training folds.  That country's test pairs are then rescored by their
    assigned half-model; every other score is unchanged.
    Needs work/ce_pairs_test_p3.npy (ce_export.py p3).  Writes work/ce_<country>_model_{A,B}.pt and
    work/ce_<country>_logit_test.npy (read by final2.py infer --ce-name ce_<country>)."""
    name = f"ce_{country.lower()}"
    Pt, text_te = load_pairs("test")
    E = pd.read_parquet(config.work("ce_pairs_test.parquet"), columns=["q_id", "p2"])
    q_id = E.q_id.values[E.p2.values < P2_MAX]                  # same filter and order as load_pairs
    cty = pd.read_csv(config.src_path("test", 1), sep="\t", dtype=str, usecols=["entity_id", "country"])
    mine = cty.set_index("entity_id").country.reindex(q_id).values == country
    p3 = np.load(config.work("ce_pairs_test_p3.npy"))[Pt.k.values]
    pseudo = np.flatnonzero(mine & ((p3 >= PSEUDO_HI) | (p3 <= PSEUDO_LO)))
    print(f"{country}: {mine.sum():,} scored test pairs, pseudo-labelled {len(pseudo):,} "
          f"({(p3[pseudo] >= PSEUDO_HI).sum():,} match / {(p3[pseudo] <= PSEUDO_LO).sum():,} non-match)", flush=True)
    Ptr, text_tr = load_pairs("train")
    text, off = np.concatenate([text_tr, text_te]), len(text_tr)
    out = np.load(config.work(f"ce{TAG}_logit_test.npy"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(0)
    for h, (folds, _) in HALVES.items():
        w = config.work(f"{name}_model_{h}.pt")
        if os.path.exists(w):
            model = CrossEncoder().to(DEV)
            model.load_state_dict(torch.load(w, map_location=DEV))
        else:
            pick = rng.choice(pseudo, min(N_ADAPT, len(pseudo)), replace=False)
            lab = sample_train(Ptr, folds, N_ADAPT, {"A": 3, "B": 4}[h])
            A = pd.DataFrame({"qi": np.r_[Pt.qi.values[pick] + off, Ptr.qi.values[lab]],
                              "pi": np.r_[Pt.pi.values[pick] + off, Ptr.pi.values[lab]],
                              "y": np.r_[(p3[pick] >= PSEUDO_HI).astype(np.int8), Ptr.y.values[lab]]})
            model, _ = fit(A, text, rng.permutation(len(A)), {"A": 5, "B": 6}[h],
                           init=config.work(f"ce{TAG}_model_{h}.pt"), lr=ADAPT_LR)
            torch.save(model.state_dict(), w)
        m = mine & (Pt.h.values == (0 if h == "A" else 1))
        print(f"{country}: half-model {h} rescores {m.sum():,} pairs", flush=True)
        out[Pt.k.values[m]] = score(model, tok, text_te, Pt.qi.values[m], Pt.pi.values[m])
    np.save(config.work(f"{name}_logit_test.npy"), out)


def score_all(splits=("train", "test")):
    """Score every exported pair with one half-model (see module doc).  Dev-fold scores already
    present in work/ce_logit_train.npy are reused; holdout and test pairs are always (re)scored."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    models = {}
    for name in HALVES:
        m = CrossEncoder().to(DEV)
        m.load_state_dict(torch.load(config.work(f"ce{TAG}_model_{name}.pt"), map_location=DEV))
        models[name] = m
    for split in splits:
        path = config.work(f"ce{TAG}_logit_{split}.npy")
        P, text = load_pairs(split)
        n_all = len(pd.read_parquet(config.work(f"ce_pairs_{split}.parquet"), columns=["p2"]))
        full = np.load(path) if os.path.exists(path) else np.full(n_all, np.nan, np.float32)  # NaN = not scored
        fold = P.fold.values if split == "train" else np.full(len(P), -1)
        half = np.where(np.isin(fold, HALVES["A"][1]), "A", np.where(np.isin(fold, HALVES["B"][1]), "B",
                                                                      np.where(P.h.values == 0, "A", "B")))
        todo = ~np.isin(fold, (1, 2, 3, 4)) | np.isnan(full[P.k.values])
        for name in HALVES:
            m = todo & (half == name)
            print(f"{split}: model {name} scores {m.sum():,} pairs", flush=True)
            full[P.k.values[m]] = score(models[name], tok, text, P.qi.values[m], P.pi.values[m])
        np.save(path, full)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if os.name == "nt":   # keep a laptop from sleeping while this process runs (reverts on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    print(f"device {DEV} | model {MODEL}", flush=True)
    args = [x for x in sys.argv[2:] if not x.startswith("--")]
    {"pilot": lambda: pilot(tuple(args) or (MODEL,), "--light" in sys.argv),
     "train": lambda: train_half(args[0]), "score": lambda: score_all(tuple(args) or ("train", "test")),
     "pack": pack, "adapt": lambda: adapt(args[0])}[cmd]()
