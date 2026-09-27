"""End-to-end pipeline (final = shift-robust v3 + house-number relation features + cross-encoder, threshold 0.9):
data -> normalization -> blocking -> matching -> output.

  python main.py all        # everything below, in order
  python main.py prep       # folds, transliteration dictionary (dev folds only), normalization
  python main.py cands      # candidate generation (train + test)
  python main.py features   # stage-1 ranker + top-25 pruning + pair/context features, IDF-weighted similarities
  python main.py validate   # stage-2 (composition-invariant) and stage-3 (stacking) out-of-fold on dev folds
  python main.py ce         # cross-encoder (needs a CUDA GPU): pair export, 2-way cross-fitted fine-tuning,
                            # scoring, stage 3 with the CE logit (out-of-fold on dev folds)
  python main.py holdout    # score on the holdout fold (fold 0)
  python main.py fit        # final stage-2 + stage-3 models on all training data
  python main.py infer      # test inference -> output/matching_results.tsv, output/candidate_pairs.tsv + validator
  python main.py adapt      # cross-encoder self-training for the country absent from training (France, GPU),
                            # then test inference again with the adapted scores -> final output/*.tsv
"""
import subprocess
import sys

POLICY = "thr0.9"   # robust to near-copy distractor density (E1 shift suite + leaderboard confirmation)
S2, TAG0, TAG = "inv3n", "s3inv3n", "s3inv3n_ce"   # TAG0: stage 3 without the CE (defines the CE pair subset)


def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main(stage):
    if stage in ("prep", "all"):
        sh("prep.py")
    if stage in ("cands", "all"):
        sh("candidates.py", "train")
        sh("candidates.py", "test")
    if stage in ("features", "all"):
        sh("train.py", "--tag", "base")                    # builds train features + stage-1 model; logs the v1 baseline
        sh("features_idf.py", "train", "train_feat")
        sh("features_num.py", "train", "train_feat")          # house-number relation features (E1)
    if stage in ("validate", "all"):
        # composition-invariant stage 2: drop raw ambiguity counts, raw competition gaps/ranks and stage-1 score;
        # add country-relative IDF similarities except the two scale-shifted max-unmatched-token weights (A8)
        # plus house-number relation features (N)
        sh("exp.py", "--tag", S2, "--train", "normal", "--evals", "normal",
           "--invariant", "size+comp+s1", "--extra", "A8N")
        sh("final2.py", "stack", "--s2", S2, "--tag", TAG0)
    if stage in ("ce", "all"):
        sh("final2.py", "fit", "--s2", S2, "--tag", TAG0)   # full stage-2 model: selects the test pairs the CE scores
        sh("ce_export.py", "export")
        sh("ce_train.py", "train", "A")                     # folds 1-2 -> scores folds 3-4
        sh("ce_train.py", "train", "B")                     # folds 3-4 -> scores folds 1-2
        sh("ce_train.py", "score")                          # holdout / test pairs: A or B by S1 hash
        sh("final2.py", "stack", "--s2", S2, "--tag", TAG, "--ce")
    if stage in ("holdout", "all"):
        sh("final.py", "holdout", "--tag", TAG, "--policy", POLICY)
    if stage in ("fit", "all"):
        sh("final2.py", "fit", "--s2", S2, "--tag", TAG)
    if stage in ("infer", "all"):
        sh("final2.py", "infer", "--tag", TAG, "--policy", POLICY, "--adapt", "--final")
    if stage in ("adapt", "all"):
        # pseudo-labels = the confident test probabilities written by infer; method validated by ce_loco.py
        # (India held out as the unseen country: +0.0121 macro F0.5)
        sh("ce_export.py", "p3")
        sh("ce_train.py", "adapt", "France")
        sh("final2.py", "infer", "--tag", TAG, "--policy", POLICY, "--adapt", "--final", "--ce-name", "ce_france")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
