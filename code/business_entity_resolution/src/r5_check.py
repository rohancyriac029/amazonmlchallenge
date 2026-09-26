"""R5: does fold-averaging make test-time stage-2 probabilities look like the OOF inputs stage 3 was trained on?
Label-free: KS distance of test stage-2 probabilities (full model vs mean of fold models) to dev OOF stage-2 probabilities."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import config
import exp
import final2

spec = final2.s2_spec("inv3n")
X, meta, cols = exp.load("test_feat", spec["base_cols"], spec["extra"])
full = lgb.Booster(model_file=config.work("stage2_s3inv3n.txt")).predict(X, num_threads=config.N_JOBS)
avg = np.mean([lgb.Booster(model_file=config.work(f"s2fold_inv3n_m_{f}.txt")).predict(X, num_threads=config.N_JOBS)
               for f in (1, 2, 3, 4)], axis=0)
Ft = pd.read_parquet(config.work("train_feat.parquet"), columns=["fold"])
oof = np.load(config.work("goof_inv3n__normal.npy"))[Ft.fold.values != 0]
o2 = np.load(config.work("goof_inv3n_m__normal.npy"))[Ft.fold.values != 0]
print(f"determinism check (re-run OOF identical): max |diff| = {np.abs(oof - o2).max():.2e}")
rng = np.random.default_rng(0)
sub = lambda a: rng.choice(a, 2_000_000, replace=False)
o = sub(oof)
for name, t in (("full-data model", full), ("mean of 4 fold models", avg)):
    t_ = sub(t)
    print(f"{name:22s} KS vs dev OOF = {ks_2samp(o, t_).statistic:.4f} | share>0.9 {np.mean(t > 0.9):.4f} "
          f"(OOF {np.mean(oof > 0.9):.4f}) | share 0.1-0.9 {np.mean((t > 0.1) & (t < 0.9)):.4f} (OOF {np.mean((oof > 0.1) & (oof < 0.9)):.4f})")
print(f"test: mean |full - avg| = {np.abs(full - avg).mean():.4f}")
