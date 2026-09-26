"""Load TSVs and compute normalized representations (cached as parquet)."""
import os
import numpy as np
import pandas as pd

import config
import normalization as N

_NS = {}


def read_source(split, k):
    df = pd.read_csv(config.src_path(split, k), sep="\t", dtype=str, keep_default_na=False,
                     quoting=3, engine="c")
    df.columns = ["entity_id", "business_name", "business_address", "country"]
    df["src"] = np.int8(k)
    return df


def read_ground_truth():
    gt = pd.read_csv(config.gt_path(), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    return {r: set(x for x in m.split(",") if x) for r, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}


def _norm_chunk(args):
    names, addrs, tdict = args
    N.set_translit_dict(tdict)
    nr = [N.name_repr(x) for x in names]
    ar = [N.addr_repr(x) for x in addrs]
    return pd.concat([pd.DataFrame(nr), pd.DataFrame(ar)], axis=1)


def normalize_frame(df, tdict=None, chunk=50000):
    tdict = tdict or {}
    names = df.business_name.tolist()
    addrs = df.business_address.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], tdict) for i in range(0, len(df), chunk)]
    parts = config.parallel_map(_norm_chunk, jobs, "_NS", {})
    out = pd.concat(parts, ignore_index=True)
    out.index = df.index
    return pd.concat([df, out], axis=1)


def load_split(split, tdict=None, tag="v1", force=False):
    """Return one DataFrame with all three sources of a split, normalized."""
    path = config.work(f"{split}_{tag}.parquet")
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    dfs = [read_source(split, k) for k in (1, 2, 3)]
    df = pd.concat(dfs, ignore_index=True)
    df = normalize_frame(df, tdict)
    df.to_parquet(path, index=False)
    return df
