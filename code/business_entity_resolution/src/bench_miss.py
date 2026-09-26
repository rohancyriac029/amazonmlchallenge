import numpy as np, pandas as pd, scipy.sparse as sp, sys
import config, blocking
from data_loader import read_ground_truth
from sparse_dot_topn import sp_matmul_topn
df = pd.read_parquet(config.work("train_v1.parquet")); gt = read_ground_truth()
country = sys.argv[1]
g = df[df.country == country]
q = g[g.src == 1].sample(20000, random_state=1); p = g[g.src != 1]
qid = q.entity_id.values; pid = p.entity_id.values
true = {(a, b) for a in q.entity_id for b in gt[a]}
mats = {ch: blocking.tfidf_pair(blocking.build_feature_lists(q, ch), blocking.build_feature_lists(p, ch), 10000) for ch in ("name", "addr")}
w = .5
cQ = sp.hstack([mats["name"][0]*np.sqrt(w), mats["addr"][0]*np.sqrt(1-w)]).tocsr(); cP = sp.hstack([mats["name"][1]*np.sqrt(w), mats["addr"][1]*np.sqrt(1-w)]).tocsr()
U = set()
for Q, P in (mats["name"], mats["addr"], (cQ, cP)):
    C = sp_matmul_topn(Q, P.T.tocsr(), top_n=30, threshold=1e-6, n_threads=8).tocoo(); U |= set(zip(qid[C.row], pid[C.col]))
miss = sorted(true - U)
print("missed", len(miss), "of", len(true))
R = df.set_index("entity_id")
# score of the missed pair in combo vs the 30th best
pos_q = {x: i for i, x in enumerate(qid)}; pos_p = {x: i for i, x in enumerate(pid)}
cols = ["business_name", "business_address", "n_core", "a_norm"]
import collections
stats = collections.Counter()
for a, b in miss[:4000]:
    i, j = pos_q[a], pos_p[b]
    sn = mats["name"][0][i].multiply(mats["name"][1][j]).sum(); sa = mats["addr"][0][i].multiply(mats["addr"][1][j]).sum()
    stats["name0"] += sn == 0; stats["addr0"] += sa == 0; stats["both0"] += (sn == 0) & (sa == 0); stats["p_addr_empty"] += R.at[b, "a_empty"]
print(dict(stats), "(of", min(len(miss), 4000), ")")
for a, b in miss[:40]:
    i, j = pos_q[a], pos_p[b]
    sn = mats["name"][0][i].multiply(mats["name"][1][j]).sum(); sa = mats["addr"][0][i].multiply(mats["addr"][1][j]).sum()
    print(f"\n{a} | {R.at[a,'business_name']} | {R.at[a,'business_address']}\n{b} | {R.at[b,'business_name']} | {R.at[b,'business_address']}  [sn={sn:.2f} sa={sa:.2f}] core='{R.at[b,'n_core']}'")
