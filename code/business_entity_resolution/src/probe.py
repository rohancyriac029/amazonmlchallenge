"""Write a leaderboard probe from existing test probabilities with a different decision rule."""
import sys, os, numpy as np, pandas as pd
import config, decision
from submission import write_sets, self_check
name, rule = sys.argv[1], sys.argv[2]
prob_file = sys.argv[3] if len(sys.argv) > 3 else "test_prob.npy"   # e.g. test_prob_s3inv3.npy
meta = pd.read_parquet(config.work("test_feat.parquet"), columns=["q", "p"])
meta["prob"] = np.load(config.work(prob_file))
df = pd.read_parquet(config.work("test_v1.parquet"), columns=["entity_id", "src", "country"])
s1 = df.entity_id.values[(df.src == 1).values]
if rule.startswith("thr"):
    sel = decision.threshold_policy(meta, float(rule[3:]))
elif rule.startswith("gate"):
    sel = decision.gated_expected_f_policy(meta, gate=float(rule[4:]), miss_mass=0.3)
pred = decision.to_sets(df, sel, s1)
out = config.work(f"probe_{name}")
os.makedirs(out, exist_ok=True)
mp, cp = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
write_sets(mp, s1, pred, "matched_entity_ids")
write_sets(cp, s1, decision.to_sets(df, meta, s1), "candidate_entity_ids")
self_check(mp, cp, s1, df.entity_id.values[(df.src != 1).values])
n = pd.Series([len(pred[i]) for i in s1]); c = df.set_index("entity_id").country.reindex(s1).values
print("matches/S1:", n.groupby(c).mean().round(3).to_dict(), "empty:", (n == 0).groupby(c).mean().round(4).to_dict())
