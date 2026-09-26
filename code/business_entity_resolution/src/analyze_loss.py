"""Loss decomposition on dev folds for an OOF run."""
import sys, numpy as np, pandas as pd
import config, decision, train
from data_loader import read_ground_truth
from evaluation import evaluate_predictions
tag, pol = sys.argv[1], sys.argv[2]
gt = read_ground_truth()
owner = {o: s for s, v in gt.items() for o in v}
F = pd.read_parquet(config.work("train_feat.parquet"), columns=["q", "p", "y", "fold", "p_a_empty", "n_cand_p"])
F["prob"] = np.load(config.work(f"oof_{tag}.npy"))
df = pd.read_parquet(config.work("train_v1.parquet"), columns=["entity_id", "src"])
ids = df.entity_id.values
dev = F.fold.values != 0
s1_all = ids[(df.src == 1).values]
from prep import fold_of
dev_ids = [x for x in s1_all if fold_of(x) != 0]
sel = train.default_policies()[pol](F)
sel = sel[sel.fold != 0]
fp = sel[sel.y == 0]
own = np.array([owner.get(x) for x in ids[fp.p.values]], dtype=object)
print(f"FP pairs {len(fp):,}: pool record unowned {np.mean(own == None):.3f}, owned by other S1 {np.mean(own != None):.3f}")
print(f"  FP n_cand_p==1: {np.mean(fp.n_cand_p.values == 1):.3f}")
# ceiling: perfect classifier over candidates
C = F[dev & (F.y == 1)]
perfect = decision.to_sets(df, C, dev_ids)
print("CEILING (perfect classifier on candidates):", end=" "); evaluate_predictions(perfect, gt, ids=dev_ids)
# current
pred = decision.to_sets(df, sel, dev_ids)
print("CURRENT:", end=" "); r = evaluate_predictions(pred, gt, ids=dev_ids)
# remove FP on unowned records (oracle) -> how much is 'unowned-copy' noise worth
keep = sel[(sel.y == 1) | pd.Series([owner.get(x) is not None for x in ids[sel.p.values]], index=sel.index)]
print("ORACLE drop FPs on unowned records:", end=" "); evaluate_predictions(decision.to_sets(df, keep, dev_ids), gt, ids=dev_ids)
keep2 = sel[(sel.y == 1) | pd.Series([owner.get(x) is None for x in ids[sel.p.values]], index=sel.index)]
print("ORACLE drop FPs on owned records:", end=" "); evaluate_predictions(decision.to_sets(df, keep2, dev_ids), gt, ids=dev_ids)
# oracle add rejected true candidates with p_a_empty
add = pd.concat([sel, F[dev & (F.y == 1) & (F.p_a_empty == 1)]]).drop_duplicates(["q", "p"])
print("ORACLE add missed with empty addr:", end=" "); evaluate_predictions(decision.to_sets(df, add, dev_ids), gt, ids=dev_ids)
add2 = pd.concat([sel, F[dev & (F.y == 1) & (F.p_a_empty == 0)]]).drop_duplicates(["q", "p"])
print("ORACLE add missed with addr:", end=" "); evaluate_predictions(decision.to_sets(df, add2, dev_ids), gt, ids=dev_ids)
