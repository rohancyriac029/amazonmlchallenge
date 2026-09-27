# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We built a blocking + learned-matching pipeline with three stages:

1. **Retrieval.** Multi-channel sparse TF-IDF retrieval, blocked by country.
2. **Stage-1 ranker.** Prunes candidates to 25 per Source-1 entity.
3. **Two LightGBM pair classifiers.** Stage 2 scores each pair. Stage 3 re-scores pairs using set-level evidence: sibling agreement between candidates, and competition between S1 entities for the same record.

A precision-first set-selection rule turns scores into matches. It maximises expected F0.5, gates out weak entities, and enforces that each S2/S3 record belongs to at most one S1 (a structural fact we measured in the ground truth).

Key techniques:

- rule-based Indic→Latin transliteration, plus a translation dictionary learned from training pairs
- name×locality conjunction blocking keys
- stacking with graph-style consistency features
- pseudo-label adaptation for countries that are unseen in training (France)
- a fine-tuned multilingual **cross-encoder** (`multilingual-e5-small`, MIT, 118M parameters) whose pair score feeds stage 3

The final model (**v2**) is additionally made **robust to dataset shift**. A train-vs-test domain classifier showed that features encoding the *composition* of a dataset do not transfer to test. These are raw name-frequency counts, raw competition gaps between S1 entities, and the stage-1 score that folds both in. Stage 2 drops them and uses country-relative IDF-weighted similarities instead. Competition between S1 entities is re-introduced only in stage 3, through probabilities.

Results:

| | v1 | v2 | v3 @ thr 0.9 | v3 + house-number @ thr 0.9 | **+ cross-encoder @ thr 0.9 (final)** |
|---|---|---|---|---|---|
| Public leaderboard (macro F0.5) | 0.960 | 0.974 | 0.975 | 0.977 | **0.983** |
| Holdout fold (training distribution) | 0.9840 | 0.9841 | 0.9810 | 0.9829 | **0.9884** |
| Dev 4-fold OOF | 0.9830 ± 0.0001 | 0.9833 ± 0.0001 | 0.9805\* | 0.9824\* | **0.9881**\* |

\* Under the 0.9 threshold. With the expected-F rule, the same models score 0.9828 (v3), 0.9842 (house-number) and 0.9884 (final) on dev. The threshold deliberately trades in-distribution recall for robustness to near-copy distractors (§5.6). Seed noise on dev is ±0.00007 (3 seeds, §5.8).

The last step (§5.8) added **house-number relation features**. The model can now tell *neighbouring premises* (a distractor a few doors away, typically on the other side of the street) from a *corrupted house number* (a typo or truncation on a true match). That cut house-number misses by 34% and the related false merges by 22%.

The final step (§5.10) added a **cross-encoder**: a small multilingual transformer fine-tuned on the supplied training pairs, which reads both records' raw text jointly. Its score is one extra stage-3 feature. Cross-fitting keeps it out-of-fold. It cut dev false merges by 60% and missed matches by 20%, and raised the leaderboard from 0.977 to **0.983**, the same size as its holdout gain.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA on all 25M records)

| | train S1 | train S2 | train S3 | test S1 | test S2 | test S3 |
|---|---|---|---|---|---|---|
| rows | 2,206,821 | 5,034,616 | 5,285,603 | 1,732,544 | 4,887,273 | 5,082,316 |
| countries | US 60%, India 40% | same | same | India 47%, US 38%, **France 15%** | | |
| empty address | 0% | 3.4% | 3.3% | 0% | 2.7% | 2.7% |
| non-ASCII names | 0% | 15.2% | 11.5% | 2.4% | 19.0% | 14.5% |

Ground-truth structure (train):

- **Singletons: 5.58%** of S1 have no match (US 5.58%, India 5.59%). The mean number of matches is 3.46; the full distribution is 0: 123k, 1: 119k, 2: 375k, 3: 531k, 4: 484k, 5: 322k, 6+: 252k (max 11).
- There are 3.69M S1→S2 and 3.94M S1→S3 links. 80.5% of S1 match both sources, 6.5% only S2, 7.5% only S3.
- **Each S2/S3 record belongs to at most one S1.** Of 7.64M matched records, 0 map to more than one S1. We exploit this as an exclusivity constraint.
- **Country always agrees** on true pairs (100.000%), so blocking by exact country string is lossless. Country is treated as an open set.
- **26% of S2/S3 records match no S1** (1.34M per source). They act as distractors, and some are near-exact copies of an S1 record.
- Name ambiguity is severe: there are only 1.54M distinct names among 2.2M S1. For example, "The Dent Tattoo" is 73 different S1 businesses, and 36% of S1 share their normalized name with another S1. Name-only evidence is therefore weak, and address/locality is essential.
- Exact raw name equality holds for only 6–14% of true pairs, and exact address equality for 4–11%.

Noise patterns observed:

- **Names:**
  - written in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Gurmukhi or Oriya, including legal suffixes (प्राइवेट लिमिटेड) and spelled-out acronyms (एलएलपी = "L-L-P")
  - typos and leetspeak ("5even", "Ne0tech", "Fami1y")
  - legal-suffix add/drop/reorder, filler words ("Services", "Center", "Group")
  - website domains ("maurewilliamscolombier.com"), DBA aliases and fully unrelated trade names that share only the address
  - phone numbers and bracketed tokens
- **Addresses:**
  - reordered components, abbreviations (St/Street/Saint, Rd, Ave)
  - wrong ordinals ("45ND")
  - leading-zero numbers ("AF-0684" vs "Af-684"), "null"/"<NULL>"/"N/A" tokens
  - state names in local script, district vs city substitutions
  - truncated or altered house numbers
  - French test data additionally has "N°", "R.", "BD.", "bis", and departments instead of regions.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage learned classifier + graph/set-level stacking + precision-first set selection (hybrid).

**Core Innovations:**
1. **Name×locality conjunction retrieval channel** (hashed TF-IDF). This is the single best blocking channel (top-30 recall 0.960). It resolves the "common name + common city" ambiguity that defeats both name-only and address-only retrieval.
2. **Stage-3 set-level stacking.** A candidate is re-scored using its similarity to the S1's other confident candidates (records of one business corroborate each other) and the best probability any *other* S1 has for the same record. Gain: **+0.0045 macro F0.5 (paired-bootstrap 95% CI [+0.0044, +0.0046])**.
3. **Exclusivity-aware expected-F0.5 set selection** with a singleton gate.
4. **Validated unseen-country adaptation** via confident pseudo-labels, for France.

---

## 3. Candidate Generation (Blocking)

**Normalization (several representations per record; `normalization.py`)**

- **Names.** The pipeline builds conservative, core, sorted, compact and phonetic-skeleton forms, plus legal-form and digit tokens:
  - *Conservative*: Unicode NFKD, accent stripping, lowercase, punctuation → space.
  - *Core*: legal forms canonicalised (ltd/limited, pvt/private, inc, corp, llc, llp, sarl, sas, …) and removed, along with generic fillers.
  - *Sorted*: core tokens in sorted order.
  - *Compact*: core without spaces, which handles domains and spacing.
  - *Phonetic skeleton*: consonant skeleton with voicing merged, for Tamil-style transliteration and vowel typos.
  - Legal-form set and digit tokens are kept separately.
  - Leetspeak is repaired inside mixed alphanumeric tokens, and `@handles`, `www`/`.com` are stripped.
- **Indic transliteration.** One offset table covers all nine ISCII-derived Unicode blocks, which share a layout. It handles inherent vowels, virama, final-schwa deletion (kept after y/r/v conjuncts), and schwa deletion before independent vowels. Spelled-out acronyms are decoded by dynamic programming over English letter names ("elelpi" → "llp", "eses" → "ss").
- **Learned transliteration dictionary.** From true pairs in the training folds only (fold 0 excluded), tokens are aligned by Jaro-Winkler, giving 530 entries such as venchars→ventures, entarapraijes→enterprises, solyushans→solutions.
- **Addresses.** Canonical abbreviations (US, Indian and French street types), null-token removal, ordinal and number-word normalisation ("Tenth" → 10), leading-zero stripping, and alphanumeric splitting ("1056c" → 1056 c). Derived fields: word set, number set, first (house) number, 5–6 digit postal/PIN codes and component count.

**Retrieval channels** (`blocking.py`). All channels use hashed features (2^26 buckets), TF-IDF with IDF over query+pool, drop features with df > 10,000, L2-normalise, and take top-K by sparse dot product (`sparse_dot_topn`, 8 threads). Blocking is by exact country.

| Channel | Features | K | Top-30 recall |
|---|---|---|---|
| name | core tokens, skeleton tokens, token bigrams, compact prefix/suffix/full | 30 | 0.653 |
| addr | address words, numbers, number+word and word bigrams | 30 | 0.852 |
| combo | name ⊕ address (0.5/0.5) | 30 | 0.920 |
| **conj** | name-token × address-word / number conjunctions (+0.2 name, +0.2 addr) | 30 | **0.960** |
| gram | character 4-grams of the compact name (typos) | 20 | 0.679 |
| **union** | | ≈79/S1 | **0.9813** |

- **Stage-1 ranker** (`ranker1.py`). A LightGBM (200 trees) over the five channel scores and their within-S1 rank, max and gap, and within-record rank and gap. It is trained out-of-fold. It keeps the top 25 per S1 and retains 99.98% of retrieved true pairs (top-10 already keeps 99.94%).
- **Candidate pairs generated:** 174.0M retrieved → **55.2M** after stage 1 on train (25 per S1), with a reduction ratio of 0.9999924. The test set has 43.3M final candidate pairs (`candidate_pairs.tsv`) — see §5.3.
- **How true matches were kept:**
  - Channels were designed from miss analysis: common-locality misses led to the conjunction channel, typos to character n-grams, transliteration to the skeleton and dictionary.
  - Recall was measured per channel and for the union.
  - `max_df` was chosen on a recall/cost sweep (3k/10k/30k).
  - The final candidate recall is **0.9811 (dev), 0.9814 (holdout)**.

---

## 4. Matching Model

**Features used** (v1 stage 2: 89 features; `features.py`).

> **Final model:** stage 2 uses 85 features. That is this list *minus* the raw ambiguity counts, the raw competition gaps/ranks and the stage-1 score/rank, *plus* 8 country-relative IDF-weighted similarities (v2 had 10; v3 dropped two scale-shifted ones, §5.6) and 9 house-number relation features (§5.8). Stage 3 adds 19 stacking features (104 in total). See §5.5–5.8 for why.

- **Name features:**
  - RapidFuzz ratio, token-sort and token-set on the core form
  - partial ratio, Jaro-Winkler and normalised Levenshtein on the compact form
  - skeleton ratio and conservative-form ratio
  - token Jaccard and containment (core, skeleton and conservative sets)
  - compact equality and containment, first-token equality
  - token counts, length ratio
  - legal-form Jaccard, digit-token equality
- **Address features:**
  - ratio, token-set, token-sort and partial ratio on the normalised address
  - word Jaccard and containment, number Jaccard and containment
  - first-number equality, and each side's house number contained in the other's numbers
  - postal/PIN equality, best numeric edit similarity
  - word and number counts
  - name appearing inside the address
- **Retrieval context:** all five channel cosines, with rank, max and gap within the S1, rank and gap within the pool record, candidate counts, and the stage-1 score and rank.
- **Ambiguity statistics (unsupervised):** how many S1 records and pool records share the normalised name, the name + house number, or the address.
- **Flags:** domain-name, DBA, phone-in-name, Indic-script, empty address (both sides), component counts, source (S2/S3).
- **Country is deliberately *not* a feature**, so France uses the same model.

**Stage 3 (set-level / graph stacking; `stack.py`).** Stage-2 out-of-fold probabilities feed 19 extra features:

- **S1 profile:** rank, max, gap and sum of P within the S1; number of candidates with P ≥ 0.5 and P ≥ 0.9.
- **Competition:** best P of any other S1 for the same record, rank within the record, and the number of confident S1 claimants.
- **Sibling support:** against the S1's other confident candidates (up to 6), the maximum and mean name/address token-set similarity, the maximum of min(name, address), a P-weighted version, same house number, and same source.

**Model type:** LightGBM binary GBDT with 127 leaves, learning rate 0.08, 600 rounds, feature/bagging fraction 0.8 (MIT licence, trained from scratch, about 2×10⁵ learned split/leaf values; far below the 8B limit). Stage 3 additionally uses the logit of a cross-encoder fine-tuned from `intfloat/multilingual-e5-small` (MIT licence, 118M parameters; §5.10). No other pretrained or external model is used.

**Threshold / decision selection** (`decision.py`, chosen on dev folds only):

1. **Exclusivity:** each S2/S3 record is kept only for its highest-scoring S1. On the base model: +0.0013 (0.9762 → 0.9775 at threshold 0.5).
2. **Expected-F0.5 prefix selection per S1:** candidates are sorted by P, and the prefix k maximising 1.25·ΣP₁..ₖ / (k + 0.25·(ΣP + 0.3)) is chosen. The empty set is chosen when Π(1−P) is larger.
3. **Singleton gate:** the prediction is empty unless the best P ≥ 0.6.

We compared thresholds 0.3–0.95, expected-F with miss-mass 0/0.3/0.6 and floors 0–0.8, and gates 0.6–0.95. All policies fall within 0.9824–0.9830, so the decision layer is robust, and we picked the best (expF, gate 0.6).

---

## 5. Results & Error Analysis

### 5.1 Validation protocol

- **Folds:** assigned per S1 with md5(id) % 5.
- **Fold 0 is an untouched final holdout** (442,303 S1). It was excluded from the transliteration dictionary, from every model, feature, threshold and policy choice, and from the error analysis. It was scored exactly once, after the configuration was frozen.
- **Development:** 4-fold OOF on folds 1–4 (1,764,518 S1). We report pooled macro F0.5, per-fold mean and std, and paired bootstrap confidence intervals over S1 entities.
- **All S1 are always queried** during blocking, so within-record competition features and exclusivity see realistic density. No labels of an evaluated fold ever enter its features.
- **Our `evaluate_predictions`** reproduces the competition metric exactly: per-S1 F0.5, singletons score 1 or 0, macro average.

### 5.2 Ablation / experiment log (dev folds, 4-fold OOF)

| Experiment | Blocking | Model | Policy | Macro F0.5 | P | R | Singleton acc | Fold std |
|---|---|---|---|---|---|---|---|---|
| base, no exclusivity | 5-ch + stage-1 top-25 | LGBM stage 2 | thr 0.5 | 0.9762 | 0.9844 | 0.9619 | 0.951 | 0.0001 |
| base | same | LGBM stage 2 | thr 0.5 + excl. | 0.9775 | 0.9863 | 0.9608 | 0.958 | 0.0002 |
| base | same | LGBM stage 2 | thr 0.7 + excl. | 0.9785 | 0.9901 | 0.9527 | 0.977 | 0.0001 |
| base | same | LGBM stage 2 | expF (miss 0.3) | 0.9785 | 0.9902 | 0.9531 | 0.954 | 0.0001 |
| **stack1** | same | **+ stage 3 stacking** | thr 0.7 | 0.9828 | 0.9915 | 0.9626 | 0.984 | 0.0002 |
| **stack1** | same | + stage 3 stacking | **expF + gate 0.6** | **0.9830** | 0.9918 | 0.9627 | 0.978 | 0.0001 |

- **Stacking vs. base:** paired bootstrap Δ = **+0.00448** [95% CI +0.00440, +0.00456].
- **Blocking ablation** (recall on a 40k-S1 sample):
  - name + addr + combo only: 0.9749 (US) / 0.9503 (India)
  - adding the conjunction and n-gram channels and the normalisation fixes: **0.9830 / 0.9786**
- **Stage-1 pruning:** learned ranker top-25 keeps 99.98% of retrieved positives, vs. max-reciprocal-rank fusion top-25 at 99.1%.

**Leave-one-country-out** (proxy for unseen France; stage 2 only; macro F0.5 on the held-out country):

| Train on → score on | Source-only | + pseudo-label adaptation | In-distribution reference |
|---|---|---|---|
| US → India | 0.9349 | **0.9404** (+0.0055) | 0.9750 |
| India → US | 0.9660 | – | 0.9808 |

The pseudo-labels on the unseen country were clean: positives at 0.990 precision and negatives at 0.9987 purity. We therefore apply the adaptation automatically to any test country absent from training.

### 5.3 Final results

- **Holdout (untouched, scored once), macro F0.5 = 0.9840:**
  - precision 0.9927, recall 0.9637, singleton accuracy 0.9823, candidate recall 0.9814
  - US 0.9855, India 0.9818
  - S2 part 0.9777, S3 part 0.9765
  - multi-match S1 0.9869, ambiguous-name S1 0.9776
- **Dev OOF:** 0.9830 (fold std 0.0001).
- **Test set, v1** (no labels; sanity statistics only):
  - Candidates: 136.9M retrieved → 43.3M final (25 per S1) in `candidate_pairs.tsv`. France has 81 retrieved candidates per S1, the same density as US (75) and India (81).
  - Predictions: 6,003,813 matches in total. Mean matches per S1: US 3.54, India 3.42, France 3.43. The predicted-empty (singleton) rate is 5.5%, 5.7% and 5.2% respectively, consistent with the 5.6% singleton rate seen in training.
  - France (unseen) used pseudo-label adaptation: 6.49M pairs, 804,611 pseudo-positives and 5,289,559 pseudo-negatives.
  - The official `utils/validate_submission.py`, including `--check-ids`, returns **PASS**.
- **Test set, v2:**
  - Same 43.3M candidates (`candidate_pairs.tsv` is byte-identical to v1).
  - 5,795,489 matches. Matches per S1: US 3.40, India 3.36, France 3.17. Predicted-empty rate: 5.7%, 5.8% and 6.1% respectively.
  - France adaptation: 750,103 pseudo-positives and 5,293,882 pseudo-negatives.
  - Validator **PASS** with `--check-ids`.
  - Public leaderboard **0.974**.
- **Test set, v3 + threshold 0.9:**
  - 5,743,719 matches. Matches per S1: US 3.36, India 3.31, France 3.22. Predicted-empty rate: 6.0%, 6.2% and 6.1% respectively.
  - Validator **PASS** with `--check-ids`.
  - Public leaderboard **0.975**.
- **Test set, final (v3 + house-number features, threshold 0.9):**
  - 5,765,771 matches. Matches per S1: US 3.36, India 3.32, France 3.29. Predicted-empty rate: 6.0%, 6.2% and 6.0% respectively.
  - France adaptation: 758,147 pseudo-positives and 5,303,057 pseudo-negatives.
  - Validator **PASS** with `--check-ids`.
  - Public leaderboard **0.977**.

### 5.4 Error analysis (dev folds)

Loss decomposition with oracles: the current score is 0.9785 (base), and a perfect classifier on our candidates would reach 0.9942.

| Error source | Oracle gain |
|---|---|
| Rejected true pairs whose record has an address | +0.0068 |
| Rejected true pairs whose record has no address | +0.0043 |
| False merges onto records with **no** ground-truth owner | +0.0037 |
| False merges onto records owned by another S1 | +0.0010 |
| Blocking misses | 0.0058 |

- **Common false positives (wrong merges):**
  - 78% of false merges are onto S2/S3 records that the ground truth assigns to **no** S1, yet which are near-exact copies of the S1 ("Inc. Roos & Thompson Clinic | 23289 Aberdeen Court, Foley" for S1 "Roos & Thompson Clinic | 23289 Aberdeen Court, Foley, AL"). They include apparent singletons with an identical record in S3. These look like distractors or label noise and are essentially indistinguishable.
  - The remainder are same-name branches with near-identical addresses (e.g. professional practices at "6721 Tower Drive" with a changed city).
- **Common false negatives (missed matches):**
  - (i) Records with an empty address whose name is shared by many S1 (name-only evidence cannot be trusted when "Bison PC" exists dozens of times).
  - (ii) Altered house numbers on otherwise identical records ("1325" → "6325 Trailridge Rd"), which the model learned to treat as distractor-like.
  - (iii) Unrelated trade names that share only the address ("Noviariax" at the S1 address).
  - (iv) Indic-script names with partial addresses.
  - Stage-3 stacking recovered a large part of (ii) and (iii) through sibling agreement: recall rose from 0.953 to 0.963 while false merges fell from 32.1k to 21.5k.

## 5.5 Generalization to the test distribution (final model v2)

**Symptom.** v1 scored 0.984 on our holdout but **0.960** on the public leaderboard.

**Diagnosis.** Every step below used unlabeled test data only; nothing was tuned on test labels.

1. **Country is not the cause.** Label-free confidence profiles show that France, US and India on test all have about 2× more uncertain candidates per S1 than the holdout (0.34–0.44 vs 0.20–0.22). The gap is test-wide, not France-specific.
2. **Extra near-copies.** Test has about 0.6 more near-identical candidates per S1 (name ≥ 90 and address ≥ 85) than train, in every country. There are also 5.75 S2/S3 records per S1 in test vs 4.68 in train.
3. **Direction check.** The same scores with a stricter 0.9 threshold scored **0.967** on the leaderboard. So v1 over-accepted on test.
4. **Cause.** A train-vs-test **domain classifier** separates candidate pairs with AUC 0.88 (US) / 0.86 (India). Its top features are all *composition-dependent*:
   - raw counts of how many records share a name, which fall from 32.6 to 17.1 for US
   - raw competition gaps and counts between S1 entities for the same record
   - the stage-1 score, which is built from them
5. **Dropping them lowers the shift.**

   | Feature set | Domain AUC, US | Domain AUC, India |
   |---|---|---|
   | full | 0.877 | 0.856 |
   | − raw counts | 0.815 | 0.790 |
   | − raw counts − competition | 0.750 | 0.728 |
   | − counts − competition − stage-1 score | 0.673 | 0.634 |

**What did not help** (measured, kept out):

- **S1-dropout** simulating test density: +0.0005 on dense dev, −0.0002 on normal dev.
- **A "smaller universe" simulation:** it made train *less* like test (domain AUC rose to 0.96).
- **EM prior-shift correction:** it recovered the simulated prior exactly, but gained only +0.0001, because the problematic pairs are not low-probability ones.

**Fix (v2).**

- **Stage 2 uses only composition-invariant inputs:** pairwise string/number similarities, retrieval scores and within-S1 ranks, plus **country-relative IDF-weighted similarities**. The added similarities are weighted Jaccard, coverage, soft-TF-IDF with Jaro-Winkler ≥ 0.88, and the weight of the most distinctive unmatched token. The IDF is computed from each country's own unlabeled records and normalized by log N, so common words in any language are down-weighted automatically.
  - Out-of-fold dev cost of the invariant set: −0.0050.
  - The IDF similarities recover +0.0017 of it (CI [+0.0017, +0.0018]).
- **Stage 3 re-introduces competition only through probabilities** (the best probability of another S1 for the same record, and sibling support). With stacking, v2 reaches **dev 0.9833 vs 0.9830 for v1** (paired bootstrap +0.0003, CI [+0.0002, +0.0004]). There is no in-distribution cost.
- **Behaviour on test.** v2 is more conservative *by itself*, with the decision rule unchanged (expF + gate 0.6): 5.80M matches vs 6.00M for v1. It scored **0.974** on the public leaderboard.

**Holdout note.** The holdout fold was scored once for v1 (0.9840) and a second time for the final v2 (0.9841). No v2 design decision used it. The v2 decisions came from dev folds and label-free test diagnostics, plus two coarse leaderboard direction checks (strict-threshold probe, final v2).

## 5.6 Audit follow-up: final model v3 + threshold 0.9

An external generalization audit made three main points: dev/holdout validation was blind to the leaderboard gap; composition dependence might remain in v2; and a shift-simulation suite should gate further changes. We implemented the items consistent with our guidelines.

**Diagnostics** (no retraining, no holdout labels):

- **Record IDs carry no signal.** The Spearman correlation between an S1's ID and its matches' IDs is −0.001, the same as random pairs.
- **In-distribution calibration is excellent.** ECE is 0.0002 overall and at most 0.002 by stratum. S1s with names shared by ≥10 others are slightly over-confident in the mid-range (0.359 predicted vs 0.329 actual).
- **Domain classifier on the full v2 model** (train vs test, including stage-3 features):

  | Feature set | Domain AUC, US | Domain AUC, India |
  |---|---|---|
  | all 97 features | 0.946 | 0.937 |
  | − `idf_n_miss_maxw`, `idf_a_miss_maxw` | **0.774** | **0.727** |
  | − all 10 IDF features | 0.767 | 0.725 |
  | − stage-3 density aggregates | 0.945 | 0.936 |
  | − within-S1 ranks as well | 0.939 | 0.932 |

  About 80% of the separability came from the two "maximum unmatched-token weight" features. Normalized IDF maps rare-token document frequencies to discrete levels that depend slightly on dataset size (0.957 in train vs 0.955 in test for a token seen once), so trees split between levels that move. Partial dependence also showed one of them *raising* match probability, which is counter-intuitive. Stage-3 aggregates and within-S1 ranks contributed ≤ 0.01 AUC and were kept.

**v3 = v2 without the two features.**

- Dev (stage 3): 0.9828 vs 0.9833 for v2, a −0.0005 cost at the audit's acceptance bar.
- Public leaderboard: 0.974, the same as v2.
- On test, v3 changes France most: 3.30 vs 3.17 matches per S1.

**E1 shift suite** (`sim_shift.py`, `sim_eval.py`).

- *The recipe was measured from training data only.* Real unowned near-copy records differ from the S1 they resemble as follows:

  | Difference from the S1 | Unowned near-copy | True match |
  |---|---|---|
  | house number differs | 90% | 15% |
  | name has an extra token | 47% | 13% |
  | fully identical | 0.9% | 18% |

  Train has 0.62 unowned near-copies per S1; test has about +0.6 more (label-free estimate).
- *Construction.* We injected Poisson(0.6) perturbed copies of each fold-4 S1's own true records: house number offset in 90% of copies, and an extra token (sampled from real distractor tokens) in 47%. All parameters were fixed before scoring. We then re-ran blocking, stage 1 and features, and scored models trained on folds 1–3.
- *Validity.* Uncertain candidates per S1 rose from 0.21 to 0.38, against test's 0.34–0.44.

| Variant | Clean fold 4 | Injected fold 4 | Leaderboard |
|---|---|---|---|
| v1 expF | 0.9831 | 0.9224 | 0.960 |
| v1 thr 0.9 | 0.9808 | 0.9311 | 0.967 |
| v2 expF | 0.9833 | 0.9239 | 0.974 |
| v2 thr 0.9 | 0.9815 | 0.9329 | – |
| v3 expF | 0.9828 | 0.9241 | 0.974 |
| v3 thr 0.9 | 0.9808 | 0.9329 | **0.975** |

- **What the suite reproduces:** the near-copy mechanism (a stricter threshold helps under distractors, and v1 < v1@0.9).
- **What it does not reproduce:** v2 > v1@0.9. The suite does not change dataset composition or include France, so it cannot judge composition-level fixes. Under our pre-registered rule it was therefore not used to choose between v2 and v3.

**Final decision.**

- v3 was adopted because it matched v2 on the leaderboard (a pre-declared confirmation submission) with much lower train-test shift.
- The 0.9 acceptance threshold was adopted because three independent signals agree: the shift suite (+0.009 for every model), the earlier v1 probe (+0.007) and the confirmation submission (+0.001 for v3).
- *Caveats.* The +0.001 on the public leaderboard alone is within noise. The threshold is density-specific: it costs about 0.002 on training-like data (holdout 0.9810 vs 0.9828 with expected-F), a deliberate trade-off.
- *Leaderboard use, in total:* five submissions — v1, the v1 strict probe, v2, and the v3 / v3@0.9 pair. No threshold sweep was run against the leaderboard.

**Audit items not implemented:**

- The per-fold transliteration dictionary. This is a correct, small dev-only bias, and fixing it would mean rebuilding all features four times. The holdout is unaffected.
- Relative `max_df` blocking and pseudo-label hardening. Both need a full rebuild, with uncertain payoff here.
- All leaderboard-fitting (Tier 3) ideas.

**Holdout use.** Fold 0 has now been scored three times: v1, v2, and the final v3@0.9 (0.9810). It never drove a decision.

## 5.7 Second audit: claims verified against the implementation

A second documentation-only audit made several claims. Each was checked against the code and, where cheap, measured (`audit2.py`, `p1_shift.py`). Dev folds and unlabeled test only; the holdout was not used.

**Claims checked in the code:**

| Claim | Finding |
|---|---|
| Hashing may depend on Python `hash()` | Not the case: `pd.util.hash_array` uses a fixed-key SipHash, so hashing is deterministic across processes. |
| The gate should be removed from the final path | Already absent: the final rule is a plain 0.9 threshold plus exclusivity. |
| House-number "conflict" and "missing" are indistinguishable | Partly: per-side number counts let trees separate them. |
| "12 bis" collides with "12" | No such rule; "bis" stays as a token. |
| Stage 3 on test uses stage-2 probabilities from the full-data model, not out-of-fold ones | True. This is a small train/test mismatch and was left unchanged. |
| (second mismatch) France's stage-2 probabilities are pseudo-label-adapted, but stage 3 was trained on unadapted out-of-fold probabilities | True. A second train/test mismatch, limited to France. Noted, not fixed. |
| Pseudo-label thresholds (0.97 / 0.03) might have been selected on results | They were fixed before any simulation. Adaptation is a single pass on stage 2 only. |

**Measured results:**

- **X1: is the shift suite's dose too high?** Label-free distractor share among near-copies, from the house-number and extra-token signatures:
  - *Method validated on train:* estimates of 0.204 and 0.193 vs 0.193 actual.
  - *Test:* 0.314 / 0.345, i.e. about 1.18 distractors per S1 vs 0.65 in train.
  - *So the extra is about 0.52 per S1.* The suite's dose of 0.6 was about right, and **the claim that 0.9 is far too conservative is not supported.** However, two things remain unexplained: the suite's severity (a clean-to-injected drop of about 0.06, vs a 0.01–0.024 leaderboard gap) and its model-independent threshold gain (+0.009 for v1, v2 and v3 alike). X2 (stage 2 alone vs stage 2 + 3 on the injected fold) would test whether stage-3 sibling support on injected copies explains them.
  - *By country:* US π = 0.33 / 0.23 and India 0.33 / 0.46, while **France π = 0.19 / 0.26, close to train**. The extra distractors are concentrated in US and India.
  - *Caveat:* the two estimators disagree per country **in opposite directions** (house-number higher for US, extra-token higher for India). Their pooled agreement (0.31 vs 0.35) hides this. It stays within the proposed 0.15 tolerance, but the signatures transfer less cleanly per country than overall.
  - *France:* being the least-shifted country fits the idea that 0.9 is slightly too strict there. Under the no-leaderboard-tuning rule it is left unchanged.
- **P1: stratified label-shift correction.** Tested on the suite with every setting fixed in advance (see `p1_shift.py`):

  | Fold 4 | raw expF | raw thr 0.9 | P1 + expF | P1 + thr 0.9 |
  |---|---|---|---|---|
  | clean | 0.9828 | 0.9808 | 0.9828 | 0.9807 |
  | injected | 0.9241 | 0.9329 | 0.9258 | 0.9342 |

  - *Detection works:* it finds no shift on clean data and raises the π estimate from 0.20 to 0.25 on injected data.
  - *Effect:* it can't fix confident errors (distractor false merges mostly have P > 0.99), so it does not replace the threshold.
  - *Not adopted:* +0.0013 on top of 0.9 is too small to justify without more evidence.
- **X8: error decomposition of the final system (dev).**
  - Threshold 0.9 halves false merges (9,860 vs 20,986).
  - The largest remaining solvable loss is rejected true pairs *with* an address: oracle +0.0082, vs +0.0042 under expF.
  - Future gains therefore need better discrimination on altered-house-number true matches, not decision-rule changes.
- **X6: empty-set calibration.** Π(1−P) is calibrated overall (0.0564 predicted vs 0.0571 actual). It under-predicts emptiness in the middle bins (e.g. 0.11 vs 0.16), which affects about 1% of S1s, and the final threshold path doesn't use it.
- **X7: coherence.** Only 1.9% of multi-claimant records have ΣP > 1, and only 0.01% of accepted pairs have a rival claimant with P > 0.5. **Coupled or Hungarian assignment would gain nothing**, which confirms the audit's own expectation.

**Future work, prioritised.** Run X2 first. If stage-3 sibling support is confirmed to amplify injected near-copies, duplicate-discounted sibling support (P2) becomes the targeted fix for the largest remaining error class, rather than a speculative idea.

**Decision.** The final submission (v3 + threshold 0.9) is unchanged. The audit's remaining proposals (duplicate-discounted siblings, three-state field encoding, fold-averaged test-time stage 2, pseudo-label hardening) each need a retrain plus leaderboard confirmation, and have small expected gains. Under our no-leaderboard-tuning rule they are recorded as future work.

## 5.8 Roadmap follow-up: house-number relation features (final model)

**Diagnostics on the v3 @ 0.9 system** (`roadmap_diag.py`, dev folds only):

- *Where the loss sits.* Oracle gains are +0.0082 for rejected true pairs *with* an address, +0.0042 without an address, and +0.0014 for false merges onto unowned records. Of 116k lost true pairs, 114.6k were never retrieved and only 1.3k were pruned by stage 1.
- *False negatives* (172.8k): 44% no address, **27% same name with a different house number**, 12% DBA/renamed. 45% sit at P 0.5–0.9 and 55% below 0.5.
- *False positives* (9.9k): the largest named class is same name with a different house number (3.1k).
- *Blocking misses:* 59.6% have no address on the other side, 25.4% share no name token, about 7% are Indic-script.
- *How house numbers differ* (strong near-duplicates whose number differs):

  | | True match (795k) | Unowned distractor (1.18M) |
  |---|---|---|
  | Median \|Δ\| | 301 | **7** |
  | \|Δ\| ≤ 12 | 15% | **69%** |
  | Same parity | 57% | **25%** |
  | Truncation / prefix | **18%** | 3% |
  | S1's number appears elsewhere in the record | **48%** | 6% |

  Distractors are neighbouring premises; true matches differ by corruption. The only numeric similarity the model had (character-based Indel) rates 30→32 as *less* similar than 1325→6325, so it could not express this.

**E1: house-number relation features** (`features_num.py`, 9 features). All are pair-relational and three-state (agree / conflict / unknown when a side lacks a number):

- log(1+|Δ|) of the first numbers
- same parity; same length
- one-digit substitution; digit transposition; prefix / truncation
- "neighbour": the numbers differ, |Δ| ≤ 12 and the street words overlap by ≥ 50%
- log of the smallest |Δ| between the S1's number and *any* number in the other record

No dataset statistics are involved. Final stage 2 has 85 features and stage 3 has 104.

**Acceptance checks** (pre-declared in the audit):

| Check | Result |
|---|---|
| Dev, thr 0.9 | 0.98054 → **0.98239**, +0.00184 [95% CI +0.00176, +0.00192] |
| Dev, expected-F | 0.98277 → 0.98420, +0.00143 [+0.00136, +0.00150] |
| E9 seed noise (v3, seeds 42 / 7 / 13) | std 0.00007 (thr 0.9), 0.00002 (expF). The E1 gain is about 26× the seed std. |
| Targeted class | house-number FN 47,098 → 30,966 (−34%), FP 3,103 → 2,425 (−22%). No other class worsened beyond noise (e.g. DBA FN 20,014 → 19,555, Indic FN 4,563 → 4,184). |
| Train-vs-test shift | domain AUC of stage 2: US 0.6868 → 0.6872, India 0.6389 → 0.6385. The new features alone reach 0.53 / 0.52, near chance. |
| Holdout (fourth scoring; no decision used it) | 0.9810 → **0.9829**, consistent with the dev gain |
| Leaderboard (one pre-declared confirmation) | 0.975 → **0.977** |

The shift suite was not used to judge E1. Its injection recipe (±1–12 house-number offsets) mirrors the new features, so a suite gain would be partly circular.

The "other" taxonomy, stage-3 amplification and fold-averaged test-time stage 2 were tested in §5.9. **Roadmap items still not run** (future work): locality-contradiction features, corroborated name-only acceptance and the DBA evidence gate (the §5.9 diagnostics make both unpromising), and measuring blocking recovery.

## 5.9 Final roadmap round: tested, none adopted

**Refreshed diagnostics of the final model** (dev, `roadmap_diag2.py`):

- *Oracle gains.* Rejected true pairs with an address: +0.0064. Name-only: +0.0041. Unowned false merges: +0.0012.
- *FN mix* (155,096): 49% name-only, 20% same name with a different house number, 13% DBA.
- *FP mix* (9,100): 27% house number differs, 24% near-identical copies with agreeing numbers.
- *Name-only FNs are mostly irreducible.* Only 12% have a unique S1 name (vs 63% of name-only FPs). Exact-name sibling support is present for 91% of FNs and 90% of FPs, so it does not separate them.
- *DBA.* Exact-address evidence is *more* frequent among FPs (64%) than FNs (51%).

Each experiment's acceptance rule was fixed before it ran. **The leaderboard was used only as a veto, and was not needed**: no candidate change passed. R3 is a robustness check of the features already in use, and it passed:

| Experiment | Hypothesis | Result | Decision |
|---|---|---|---|
| R1: strict threshold only where house numbers conflict/unknown, calibrated expected-F where they agree | Test's extra distractors change the house number, so pairs with agreeing numbers don't need the strict threshold | Dev +0.00053 [+0.00049, +0.00057]; suite tie (0.93674 vs 0.93654). **Pre-registered precondition failed:** the label-free distractor share among number-agreeing near-copies is 0.044 on test vs 0.029 on train (the signature is exact on train), driven by India (0.12). R1's dev gain also came with more FPs in exactly that class. | **Rejected.** A US/France-only variant would be post-hoc and was not pursued. |
| R2 / X2: stage-3 sibling support amplifies near-copies | The stacking gain should shrink under injected distractors | Stacking gain at thr 0.9: clean +0.0132, injected **+0.0156** | **Refuted.** Dedup-sibling variant (and the density-feature swap) dropped. |
| R4: residual house-number audit | A secondary-number pattern could separate the remaining 31k FNs | Remaining FNs with \|Δ\| ≤ 12 are true matches with genuinely nearby numbers, indistinguishable from neighbouring-premises distractors; secondary numbers are already covered by `a_njacc`/`a_ncontain` | **No new features** (no pattern ≥ 30%) |
| R5: fold-averaged test-time stage 2 | Matching stage 3's OOF training inputs should reduce the train/test mismatch | Determinism check passed (re-run OOF identical to 6×10⁻⁸). KS to dev-OOF probabilities: full-data model 0.0519, **fold average 0.0668 (further away)** | **Rejected**; the full-data model stays |
| R3: leave-one-country-out check of the house-number features | The features should transfer to an unseen country (pre-declared: the gain must hold in both directions) | Stage 2, trained on one country and scored on the other, with vs without the 9 features. US → India: 0.9266 → **0.9323**, +0.0057 [+0.0054, +0.0059]; missed 282,611 → 251,547, false merges 83,466 → 82,691. India → US: 0.9555 → **0.9627**, +0.0073 [+0.0071, +0.0074]; false merges 82,676 → 64,361, missed 249,785 → 237,260. (The thr 0.7 policy gives +0.0062 / +0.0068.) | **Passed in both directions.** The features encode a pair relation, not a country-specific pattern, so they stay. The cross-country gain is larger than in-distribution because the weaker cross-country baseline leans more on the number relation. |

The final submission is unchanged (**public leaderboard 0.977**).

## 5.10 Cross-encoder pair model

**Motivation.** Leaderboard scores near 0.99 implied that others were extracting signal we were not. We first ruled out a data leak:

- Entity IDs of matched records are uncorrelated (Spearman 0.0001), and so is row order (0.001).
- No test record's name + address occurs in train.

So the gap had to come from modelling. Until now every model was gradient boosting on hand-built features. The rules' ≤ 8B-parameter, MIT/Apache clause allows a pretrained text model.

**Design** (`ce_export.py`, `ce_train.py`, `final2.py --ce`):

- **Model.** `intfloat/multilingual-e5-small` (MIT licence, 118M parameters, 1.5% of the 8B limit), used only as the initialisation. It is fine-tuned as a cross-encoder on the supplied training pairs: `S1 "name | address"` [SEP] `candidate "name | address"`, mean pooling, one linear logit. It reads Devanagari and Latin script natively. No external data, API or lookup is involved.
- **Pairs scored.** The CE scores a pair when its stage-2 probability is in [0.01, 0.999) and it ranks in its S1's top 8. That is 4.0M train and 3.57M test pairs, covering 97% of dev missed matches and 97% of false matches of the previous final model. Other pairs get a missing value.
- **Cross-fitting.** Model A trains on dev folds 1–2 and scores folds 3–4; model B does the reverse. Holdout and test pairs are scored by A or B, chosen by a hash of the S1 id. So every pair is scored by exactly one model that never saw its S1.
- **Training.** 1M pairs per half-model (70% from the uncertain zone 0.02 < p2 < 0.98), one epoch, learning rate 5×10⁻⁵, bf16, up to 96 tokens. On one 6 GB laptop GPU (RTX 4050): ~28 min per half-model, scoring at ~2,900 pairs/s.
- **Integration.** Stage 3 gets exactly one new feature, `ce_logit` (105 features). Stage 2 and the decision rule (thr 0.9 + exclusivity) are unchanged.

**Pilot gate** (fixed before running): a CE trained on only 200k pairs had to cut stage-3 pair errors at thr 0.9 by ≥ 2% on held-out S1s. It cut them by **16.4%** (2,026 → 1,693), and log-loss fell 0.0434 → 0.0358.

**Pre-registered acceptance checks:**

| Check | Result |
|---|---|
| Dev, thr 0.9 (bootstrap) | 0.98239 → **0.98809**, +0.00571 [95% CI +0.00562, +0.00580], about 80× the seed std |
| Dev, expected-F | 0.98420 → 0.98842, +0.00422 [+0.00414, +0.00430] |
| Error counts (dev, thr 0.9) | false merges 9,100 → **3,631** (−60%); missed 270,995 → 215,767 (−20%) |
| Per class (dev, thr 0.9, FN / FP) | Every class improves. Indic names 4,184 → 1,103 / 442 → 115. House number differs 30,966 → 8,238 / 2,425 → 839. DBA 19,555 → 13,388 / 1,050 → 509. "Other" 21,194 → 3,353 / 3,570 → 860. No-address 76,169 → 73,104 / 1,283 → 1,219. |
| Memorisation (`ce_leak.py`) | Folds split S1 entities, but an S2/S3 record can occur in a CE training pair and later as a candidate of a scored S1; test shares no records with train. Pair errors fall **57%** on records the scoring model *never saw* in training, vs 5.5% on records it saw. The gain does not come from memorisation. |
| Train-vs-test shift (`ce_domain.py`) | Rise in domain AUC over the stage-2 probability alone: US +0.022, India +0.015. **Above the pre-registered +0.01**; that limit was set against too weak a base. The trusted address similarity `a_tset` gives +0.053 / +0.024, and name similarity `n_tset` +0.025 / +0.008. The CE's shift is within the range of ordinary pair features. |
| Holdout (fifth scoring; no decision used it) | 0.9829 → **0.9884** (India 0.9879, US 0.9887), consistent with dev |
| Leaderboard (one pre-declared confirmation, veto only) | 0.977 → **0.983** (+0.006, matching the holdout gain of +0.0055). **Adopted as the final model.** |

**A construction error the shift check caught.** The first version scored holdout and test pairs with the *mean* of A and B. Each half-model saturates at its own logit plateaus, so the mean falls between them, at values stage 3 never saw in training. Dev-vs-test domain AUC rose by **+0.37 (US) and +0.43 (India)**. Rounding to the bf16 grid did not help (+0.35), which ruled out a numeric-precision explanation. Scoring each holdout/test pair with a single half-model brought the rise to the +0.02 level above. Dev results are unaffected: dev pairs were always scored by one model.

**Cost.** Two GPU runs of ~2.5 h on a laptop (free). About 3 h of EC2 CPU time for export and stage-3 evaluation; the instance was stopped while the GPU ran.

---

## 6. Conclusion

Careful blocking (name×locality conjunctions, transliteration-aware normalisation) and a learned ranker give a 0.981 recall ceiling at 25 candidates per entity. Pairwise gradient boosting, set-level stacking and an exclusivity-aware expected-F0.5 decision layer reach 0.984 on the holdout.

The largest real-world gain came from **generalisation**. We diagnosed, label-free, that composition-dependent features did not transfer to the test set. Replacing them with invariant and country-relative ones raised the public leaderboard from 0.960 to **0.974**, at no cost in-distribution. An audit-driven follow-up removed a further scale-shifted feature pair (v3) and adopted a distractor-robust acceptance threshold, supported by a measured near-copy shift simulation (0.975). A final evidence-driven step added pair-relational house-number features, which separate neighbouring premises from corrupted numbers, for **0.977**. A fine-tuned multilingual cross-encoder, one extra stage-3 feature trained with cross-fitting on a laptop GPU, gave the final **0.983**.

The main lesson: features describing *the dataset* rather than *the pair* are a hidden overfitting risk in entity resolution. A train-vs-test domain classifier is a cheap way to find them, and the same check caught a construction error in the cross-encoder's test-time scoring (§5.10) before it reached a submission.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its README for exact commands):

```
src/
  main.py            end-to-end orchestration: prep → cands → features → validate → ce → holdout → fit → infer
  prep.py            folds, transliteration dictionary (train folds only), normalization → parquet
  normalization.py   Indic transliteration, acronym decoding, name/address representations
  translit_dict.py   learned transliteration dictionary
  data_loader.py     TSV loading (sep="\t", QUOTE_NONE), parallel normalization
  eda.py             EDA + ground-truth analysis
  blocking.py        hashed TF-IDF retrieval channels (country-blocked)
  candidates.py      candidate generation, labelling, recall report
  ranker1.py         stage-1 candidate ranker (top-25)
  features.py        pair, context and ambiguity features
  train.py           stage-2/3 OOF training, policies, subset reports, experiment log
  stack.py           stage-3 set-level (graph) features
  decision.py        exclusivity, thresholds, expected-F0.5 + singleton gate
  adapt.py           unseen-country pseudo-label adaptation (+ LOCO simulation)
  final.py           holdout scoring (v1 and v2), v1 fit/inference
  final2.py          final v2: composition-invariant stage 2 + stage-3 stacking, fit, test inference
  features_idf.py    country-relative IDF-weighted similarities
  ce_export.py       cross-encoder pair subset → entity-id pair lists
  ce_train.py        cross-encoder fine-tuning (2-way cross-fitting), scoring, pilots (GPU)
  ce_domain.py       CE train-vs-test shift check;  ce_leak.py  CE memorisation check
  exp.py             generalisation experiments (invariant feature sets, LOCO, simulations)
  diag_*.py, probe.py, prior_shift.py, dense.py   label-free shift diagnostics and negative-result experiments
  evaluation.py      exact macro F0.5 + diagnostics, paired bootstrap
  error_analysis.py  error taxonomy; analyze_loss.py oracle loss decomposition; policy_sweep.py
  submission.py      TSV writer + self-check
```

Reproduce: `cd src && ER_ROOT=<student_resource> python main.py all`. This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs `utils/validate_submission.py`.

### B. Additional Results

- **Computational cost.** Measured on AWS r7i.2xlarge (8 vCPU, 64 GB RAM, plus 32 GB swap):
  - normalisation: about 5 min per split
  - train candidate generation (2.2M × 10.3M): 30 min
  - stage 1: 40 min
  - stage-2 features (55M pairs): 15 min
  - 4-fold OOF: 35 min
  - stacking features: 8 min
- **Reproducibility:** fixed seeds (42), deterministic md5 folds, pinned requirements, and all intermediate artefacts cached under `work/`.
- **License compliance:** numpy, pandas, scipy (BSD); pyarrow (Apache-2.0); lightgbm (MIT); rapidfuzz (MIT); sparse_dot_topn (Apache-2.0); scikit-learn (BSD); torch (BSD); transformers (Apache-2.0). The models are LightGBM models trained from scratch plus one cross-encoder fine-tuned from `intfloat/multilingual-e5-small` (MIT licence, 118M parameters, 1.5% of the 8B limit). No LLM is used.
- **No-external-data compliance:** apart from fetching the pretrained cross-encoder initialisation (model weights, not data; `ER_CE_MODEL` can point to a local copy), the code makes no network calls, uses no geocoding, registries or lookups, and uses no external data. It uses only the challenge TSVs. Linguistic normalisation tables (abbreviations, legal forms, letter names, the Indic script layout) are generic rules written in code. The transliteration dictionary is learned from training pairs only.
