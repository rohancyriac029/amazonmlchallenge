"""EDA + ground-truth structure analysis (raw data, no normalization needed)."""
import collections
import sys

import numpy as np
import pandas as pd

import config
from data_loader import read_source, read_ground_truth


def describe(df, name):
    print(f"\n=== {name}: rows={len(df):,}")
    print("unique ids:", df.entity_id.nunique(), "| id prefix:", df.entity_id.str[:3].value_counts().to_dict())
    print("country:", df.country.value_counts().to_dict())
    for c in ["business_name", "business_address"]:
        s = df[c]
        print(f"{c}: empty={np.mean(s.str.strip() == ''):.4f} unique={s.nunique():,} "
              f"len_mean={s.str.len().mean():.1f} len_p99={s.str.len().quantile(.99):.0f} "
              f"non_ascii={np.mean(s.str.contains(r'[^\x00-\x7f]', regex=True)):.4f}")
    print("dup name+addr:", df.duplicated(["business_name", "business_address", "country"]).sum())


def main(split="train"):
    srcs = {k: read_source(split, k) for k in (1, 2, 3)}
    for k, d in srcs.items():
        describe(d, f"{split} S{k}")
    if split != "train":
        return
    gt = read_ground_truth()
    s1 = srcs[1].set_index("entity_id")
    other = pd.concat([srcs[2], srcs[3]]).set_index("entity_id")
    print("\n=== Ground truth")
    print("GT rows:", len(gt), "| S1 rows:", len(s1), "| S1 ids in GT:", len(set(gt) & set(s1.index)))
    sizes = np.array([len(v) for v in gt.values()])
    print("singleton frac: %.4f" % np.mean(sizes == 0), "| mean matches %.3f" % sizes.mean())
    print("size dist:", collections.Counter(sizes.tolist()).most_common(15))
    n2 = np.array([sum(x.startswith("S2") for x in v) for v in gt.values()])
    n3 = np.array([sum(x.startswith("S3") for x in v) for v in gt.values()])
    print("S1->S2 total %d, S1->S3 total %d, both %.4f, only S2 %.4f, only S3 %.4f" % (
        n2.sum(), n3.sum(), np.mean((n2 > 0) & (n3 > 0)), np.mean((n2 > 0) & (n3 == 0)), np.mean((n2 == 0) & (n3 > 0))))
    owner = collections.Counter(x for v in gt.values() for x in v)
    multi = sum(1 for c in owner.values() if c > 1)
    print("matched S2/S3 records: %d, mapped to >1 S1: %d" % (len(owner), multi))
    for k in (2, 3):
        ids = set(srcs[k].entity_id)
        m = sum(1 for x in owner if x.startswith(f"S{k}"))
        print(f"S{k}: {len(ids):,} records, {m:,} matched ({m / len(ids):.4f}), unmatched {len(ids) - m:,}")
    # country agreement / exact field agreement on true pairs
    pairs = [(a, b) for a, v in gt.items() for b in v]
    pa = pd.DataFrame(pairs, columns=["s1", "o"])
    pa = pa.join(s1[["business_name", "business_address", "country"]], on="s1")
    pa = pa.join(other[["business_name", "business_address", "country"]], on="o", rsuffix="_o")
    print("true pairs:", len(pa), "| missing other rec:", pa.business_name_o.isna().sum())
    print("country agree: %.5f" % np.mean(pa.country == pa.country_o))
    print(pd.crosstab(pa.country, pa.country_o))
    low = lambda s: s.fillna("").str.lower().str.strip()
    pa["src"] = pa.o.str[:2]
    pa["name_eq"] = low(pa.business_name) == low(pa.business_name_o)
    pa["addr_eq"] = low(pa.business_address) == low(pa.business_address_o)
    pa["addr_empty_o"] = low(pa.business_address_o) == ""
    print(pa.groupby(["country", "src"])[["name_eq", "addr_eq", "addr_empty_o"]].mean())
    # singletons by country
    s1c = s1.country.to_dict()
    df = pd.DataFrame({"id": list(gt), "n": sizes})
    df["country"] = df.id.map(s1c)
    print(df.groupby("country").n.agg(["mean", lambda x: np.mean(x == 0), "count"]))
    # unmatched S2/S3 records: are they near-duplicates of matched ones?
    um = other[~other.index.isin(owner.keys())].sample(15, random_state=0)
    print("\nSample UNMATCHED S2/S3 records:\n", um[["business_name", "business_address", "country"]].to_string())
    sing = [k for k, v in gt.items() if not v][:15]
    print("\nSample SINGLETON S1 records:\n", s1.loc[sing, ["business_name", "business_address", "country"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
