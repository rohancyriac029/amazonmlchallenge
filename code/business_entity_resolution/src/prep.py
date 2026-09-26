"""Step 1: folds, transliteration dictionary (train folds only), normalization -> parquet."""
import hashlib
import os
import random
import time

import pandas as pd

import config
import translit_dict
from data_loader import read_source, read_ground_truth, load_split

N_FOLDS = 5
HOLDOUT_FOLD = 0


def fold_of(s1_id):
    return int(hashlib.md5(s1_id.encode()).hexdigest(), 16) % N_FOLDS


def main():
    t0 = time.time()
    gt = read_ground_truth()
    tpath = config.work("translit_dict.json")
    if not os.path.exists(tpath):
        s1 = read_source("train", 1).set_index("entity_id").business_name
        other = pd.concat([read_source("train", 2), read_source("train", 3)]).set_index("entity_id").business_name
        pairs = [(s1[a], other[b]) for a, v in gt.items() if fold_of(a) != HOLDOUT_FOLD for b in v]
        random.Random(config.SEED).shuffle(pairs)
        d = translit_dict.learn(pairs[:3_000_000])
        translit_dict.save(d, tpath)
        print(f"translit dict: {len(d)} entries ({time.time() - t0:.0f}s)")
        del s1, other, pairs
    d = translit_dict.load(tpath)
    for split in ("train", "test"):
        df = load_split(split, d)
        print(f"{split}: {len(df):,} rows normalized ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
