# Business Entity Resolution: reproducible pipeline

Blocking → stage-1 candidate ranker → stage-2 LightGBM pair matcher → precision-first
set selection. Everything runs offline, uses only the challenge files, and needs no
pretrained model. The only learned models are two LightGBM classifiers trained from
scratch on the provided training data.

## 1. Environment

* Python 3.12, Linux x86_64. Development and the final run used AWS EC2 r7i.2xlarge
  (8 vCPU, 64 GB RAM) plus a 32 GB swap file. About 64 GB RAM is recommended, because
  the full training candidate set has 174M pairs.
* Install the dependencies (all MIT, Apache-2.0 or BSD):

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

## 2. Data layout

The pipeline expects the challenge layout. `ER_ROOT` is the folder that contains
`dataset/` and `utils/`, i.e. the `student_resource/` directory.

```
$ER_ROOT/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
$ER_ROOT/dataset/test/test_source{1,2,3}.tsv
$ER_ROOT/utils/validate_submission.py
```

Environment variables:

* `ER_ROOT`: data root (default: three levels above `src/`)
* `ER_WORK`: cache directory (default `$ER_ROOT/work`)
* `ER_OUT`: output directory (default `$ER_ROOT/output`)
* `ER_JOBS`: number of worker processes (default: CPU count)

## 3. One-command reproduction

```bash
cd src
export ER_ROOT=/path/to/student_resource ER_JOBS=8
python main.py all
```

This produces `$ER_ROOT/output/matching_results.tsv` and `$ER_ROOT/output/candidate_pairs.tsv`,
then runs the official validator, which must print `PASS`.

## 4. Step by step

| Step | Command | What it does | Time (8 vCPU) |
|---|---|---|---|
| prep | `python main.py prep` | S1 5-fold assignment (md5 of id); transliteration dictionary from dev-fold pairs only; normalization of all records → parquet | 10 min |
| cands | `python main.py cands` | 5-channel hashed TF-IDF retrieval blocked by country (≈80 candidates per S1), train + test | 30 + 24 min |
| features | `python main.py features` | stage-1 ranker (OOF) → top-25 per S1 → pair/context features; logs the v1 baseline; IDF-weighted similarities | ≈ 2 h |
| validate | `python main.py validate` | **composition-invariant stage 2** (`exp.py --invariant size+comp+s1 --extra A8`) and **stage-3 stacking**, both 4-fold out-of-fold on dev folds | ≈ 1.3 h |
| holdout | `python main.py holdout` | score on the holdout fold 0 | 5 min |
| fit | `python main.py fit` | stage-2 and stage-3 models on all training data | 30 min |
| infer | `python main.py infer` | test: candidates/features if missing → stage 2 → pseudo-label adaptation for countries absent from training → stage 3 → decision policy → `output/*.tsv` → self-check + official validator | 30 min |

The final configuration is **v3 with an acceptance threshold of 0.9** (tags `inv3` / `s3inv3`, policy `thr0.9`). v3 is v2 without two scale-shifted IDF features. The earlier v1 (`train.py --stack`, `final.py fit/infer`) and v2 (`inv2` / `s3inv2`) configurations are kept for the ablation record.

**Analysis and diagnostic tools** (development only):

| Script | Purpose |
|---|---|
| `error_analysis.py` | error taxonomy with examples |
| `analyze_loss.py` | oracle loss decomposition |
| `policy_sweep.py` | decision-policy sweep |
| `adapt.py sim` | leave-one-country-out adaptation simulation |
| `exp.py` | generalization experiments: training tables, invariant feature sets, LOCO, bootstrap |
| `dense.py` | S1-dropout / small-universe simulations (negative results, kept for the record) |
| `prior_shift.py` | EM label-shift correction check |
| `diag_country.py`, `diag_copies.py`, `diag_copies2.py`, `diag_domain.py`, `diag_domain_sets.py`, `diag_test_examples.py` | label-free train-vs-test shift diagnostics |
| `probe.py` | builds a leaderboard probe from existing test scores with a different decision rule |
| `audit_diag.py` | audit diagnostics: ID leakage, domain AUC of feature sets, partial dependence, calibration by stratum |
| `sim_shift.py`, `sim_eval.py` | E1 shift suite: measured near-copy perturbations injected into dev fold 4, frozen-variant scoring |
| `audit2.py` | second-audit checks: label-free distractor share (X1), error decomposition (X8), empty-set calibration (X6), probability coherence (X7) |
| `p1_shift.py` | stratified label-shift correction for near-copy candidates (tested, not adopted) |
| `bench_block.py`, `bench_miss.py` | blocking recall benchmark and miss analysis |
| `eda.py` | exploratory data analysis |

## 5. Validation protocol

* Folds are assigned per S1 entity with `md5(entity_id) % 5`.
* **Fold 0 is the untouched final holdout.** It is excluded from the transliteration
  dictionary, from all model selection, threshold/policy selection and feature
  ablations. It is scored once, with `final.py holdout`.
* Development uses 4-fold out-of-fold predictions on folds 1–4. We report pooled macro
  F0.5 plus the per-fold mean and std. Experiments are compared with a paired bootstrap
  over S1 entities (`train.py --compare <tag>`).
* All S1 records are always queried during blocking, so the within-record competition
  features and the one-owner constraint see realistic density. Labels of evaluated
  folds are never used to build their features.

## 6. Source layout

```
src/
  config.py          paths, env vars, portable process-pool helper
  normalization.py   Indic→Latin transliteration, acronym decoding, name/address representations
  translit_dict.py   learns transliterated-token → English-token dictionary from training pairs
  data_loader.py     TSV reading (sep="\t", QUOTE_NONE) + parallel normalization → parquet
  prep.py            folds, dictionary, normalization
  eda.py             EDA and ground-truth structure analysis
  blocking.py        hashed sparse TF-IDF channels (name, address, combo, name×locality, char 4-gram)
  candidates.py      candidate generation, labelling, recall report
  ranker1.py         stage-1 LightGBM candidate ranker (prunes to top-25 per S1)
  features.py        pair features (rapidfuzz), context/competition features, ambiguity counts
  train.py           stage-2/3 OOF training, policies, subset reports, experiment log
  stack.py           stage-3 set-level features (S1 profile, competition, sibling support)
  adapt.py           pseudo-label adaptation for unseen countries (+ LOCO simulation)
  decision.py        threshold / expected-F0.5 set selection, one-owner exclusivity
  evaluation.py      exact competition macro F0.5 + diagnostics, paired bootstrap
  error_analysis.py  error taxonomy + examples
  final.py           holdout evaluation (both versions); v1 fit/inference
  final2.py          v2 (shift-robust) stage-3 OOF, final fit, test inference
  features_idf.py    country-relative IDF-weighted similarities (change A)
  error_analysis.py, analyze_loss.py, policy_sweep.py, bench_block.py, bench_miss.py   analysis tools
  submission.py      writes the TSVs and self-checks them
  main.py            orchestrates the full pipeline
```
