"""Competition metric: macro F0.5 over Source-1 entities (singletons included)."""
import numpy as np


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def evaluate_predictions(predictions, ground_truth, candidates=None, ids=None, verbose=True, return_scores=False):
    """predictions/ground_truth/candidates: dict s1_id -> set of ids.

    ids: the S1 entities to evaluate (defaults to ground_truth keys).
    """
    ids = list(ground_truth.keys()) if ids is None else list(ids)
    f, P, R = [], [], []
    single_ok = single_n = 0
    fm = missed = exact = 0
    n_pred = n_true = n_tp = 0
    cand_hit = cand_tot = 0
    nonsingle_f = []
    for i in ids:
        t = ground_truth.get(i, set())
        p = predictions.get(i, set())
        s = f05(p, t)
        f.append(s)
        tp = len(p & t)
        n_pred += len(p)
        n_true += len(t)
        n_tp += tp
        fm += len(p) - tp
        missed += len(t) - tp
        exact += int(p == t)
        P.append(tp / len(p) if p else (1.0 if not t else 0.0))
        R.append(tp / len(t) if t else 1.0)
        if not t:
            single_n += 1
            single_ok += int(not p)
        else:
            nonsingle_f.append(s)
        if candidates is not None and t:
            c = candidates.get(i, set())
            cand_hit += len(c & t)
            cand_tot += len(t)
    res = {
        "n": len(ids),
        "macro_f05": float(np.mean(f)) if f else 0.0,
        "macro_precision": float(np.mean(P)) if P else 0.0,
        "macro_recall": float(np.mean(R)) if R else 0.0,
        "singleton_acc": single_ok / single_n if single_n else float("nan"),
        "n_singletons": single_n,
        "nonsingleton_f05": float(np.mean(nonsingle_f)) if nonsingle_f else float("nan"),
        "false_merges": fm,
        "missed": missed,
        "exact_set_acc": exact / len(ids) if ids else 0.0,
        "n_pred": n_pred,
        "n_true": n_true,
        "n_tp": n_tp,
    }
    if candidates is not None:
        res["candidate_recall"] = cand_hit / cand_tot if cand_tot else float("nan")
        res["n_candidates"] = int(sum(len(candidates.get(i, ())) for i in ids))
    if return_scores:
        res["_scores"] = np.asarray(f, dtype=np.float32)
    if verbose:
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items() if not k.startswith("_")))
    return res


def paired_bootstrap(a, b, n_boot=1000, seed=0):
    """a, b: per-entity F0.5 arrays (same entities). Returns mean diff (b-a) and 95% CI."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
