"""Write matching_results.tsv / candidate_pairs.tsv and self-check them."""
import os

import pandas as pd


def write_sets(path, s1_ids, sets, col):
    """One row per S1 id (input order), comma-joined sorted unique ids, empty for none."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i in s1_ids:
            v = sorted(sets.get(i, ()))
            f.write(f"{i}\t{','.join(v)}\n")


def self_check(match_path, cand_path, s1_ids, pool_ids):
    s1 = set(s1_ids)
    pool = set(pool_ids)
    res = {}
    for path, col in ((match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        assert list(df.columns) == ["source1_entity_id", col], df.columns
        assert df.source1_entity_id.is_unique and set(df.source1_entity_id) == s1, "S1 coverage"
        sets = {}
        for a, b in zip(df.source1_entity_id, df[col]):
            ids = [x for x in b.split(",") if x]
            assert len(ids) == len(set(ids)), f"duplicate ids for {a}"
            assert all(x in pool for x in ids), f"unknown id for {a}"
            sets[a] = set(ids)
        res[col] = sets
    m, c = res["matched_entity_ids"], res["candidate_entity_ids"]
    assert all(m[a] <= c[a] for a in m), "matches must be subset of candidates"
    n_match = sum(len(v) for v in m.values())
    n_cand = sum(len(v) for v in c.values())
    print(f"self-check OK: {len(m):,} S1 rows, {n_match:,} matches, {n_cand:,} candidates, "
          f"{sum(1 for v in m.values() if not v):,} singletons predicted")
