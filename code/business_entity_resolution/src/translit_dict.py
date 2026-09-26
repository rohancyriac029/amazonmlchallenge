"""Learn a transliterated-token -> Latin-token dictionary from labelled training pairs.

For each true (S1, S2/S3) pair where the S2/S3 name is written in an Indic
script and the S1 name in Latin, tokens are aligned greedily by string
similarity; frequent, consistent alignments become dictionary entries
(e.g. 'venchars' -> 'ventures', 'praivet' -> 'private').  Only labels of the
training fold are used.
"""
import collections
import json

from rapidfuzz.distance import JaroWinkler

import normalization as N


def _toks(s):
    return [t for t in N._NON_ALNUM_RE.split(N.fold(s)) if t]


def learn(pairs, min_count=3, min_share=0.5, min_sim=0.55):
    """pairs: iterable of (s1_name_raw, other_name_raw)."""
    co = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for a, b in pairs:
        if not N.has_indic(b) or N.has_indic(a):
            continue
        ta, tb = _toks(a), _toks(b)
        if not ta or not tb:
            continue
        for t in tb:
            tot[t] += 1
            best, bs = None, 0.0
            for u in ta:
                s = JaroWinkler.normalized_similarity(t, u)
                if s > bs:
                    best, bs = u, s
            if best is not None and bs >= min_sim:
                co[t][best] += 1
    d = {}
    for t, c in co.items():
        u, n = c.most_common(1)[0]
        if n >= min_count and n / tot[t] >= min_share and u != t:
            d[t] = u
    return d


def save(d, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
