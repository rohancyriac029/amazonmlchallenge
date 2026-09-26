"""Sweep decision policies on dev-fold OOF scores (never the holdout)."""
import sys, numpy as np, pandas as pd
import config, decision
from data_loader import read_ground_truth
from evaluation import evaluate_predictions
from prep import fold_of
tag = sys.argv[1]
country = sys.argv[2] if len(sys.argv) > 2 else ""
gt = read_ground_truth()
F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold"])
F["prob"] = np.load(config.work(f"oof_{tag}.npy"))
df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src", "country"])
s1 = df.entity_id.values[(df.src == 1).values]
s1c = df.country.values[(df.src == 1).values]
dev = [x for x, c in zip(s1, s1c) if fold_of(x) != 0 and (not country or c == country)]
print(f"evaluating {len(dev):,} dev S1 ({country or 'all'})")
res = []
def run(name, sel):
    r = evaluate_predictions(decision.to_sets(df, sel, s1), gt, ids=dev, verbose=False)
    res.append((name, r["macro_f05"], r["macro_precision"], r["macro_recall"], r["singleton_acc"]))
    print(f"{name:40s} F={r['macro_f05']:.5f} P={r['macro_precision']:.4f} R={r['macro_recall']:.4f} single={r['singleton_acc']:.4f}", flush=True)
for t in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    run(f"thr{t}", decision.threshold_policy(F, t))
for mm in (0.3,):
    for fl in (0.0, 0.5, 0.7, 0.8):
        run(f"expF miss{mm} floor{fl}", decision.expected_f_policy(F, miss_mass=mm, t_floor=fl))
# singleton gate: empty set unless best prob >= g, then expF
for g in (0.6, 0.7, 0.8, 0.9, 0.95):
    sel = decision.expected_f_policy(F, miss_mass=0.3)
    mx = F.groupby("q").prob.max()
    sel = sel[sel.q.map(mx).values >= g]
    run(f"expF miss0.3 + gate{g}", sel)
