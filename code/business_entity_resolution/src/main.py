"""End-to-end pipeline (final = shift-robust v3 + threshold 0.9): data -> normalization -> blocking -> matching -> output.

  python main.py all        # everything below, in order
  python main.py prep       # folds, transliteration dictionary (dev folds only), normalization
  python main.py cands      # candidate generation (train + test)
  python main.py features   # stage-1 ranker + top-25 pruning + pair/context features, IDF-weighted similarities
  python main.py validate   # stage-2 (composition-invariant) and stage-3 (stacking) out-of-fold on dev folds
  python main.py holdout    # score on the holdout fold (fold 0)
  python main.py fit        # final stage-2 + stage-3 models on all training data
  python main.py infer      # test inference -> output/matching_results.tsv, output/candidate_pairs.tsv + validator
"""
import subprocess
import sys

POLICY = "thr0.9"   # robust to near-copy distractor density (E1 shift suite + leaderboard confirmation)
S2, TAG = "inv3", "s3inv3"


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
    if stage in ("validate", "all"):
        # composition-invariant stage 2: drop raw ambiguity counts, raw competition gaps/ranks and stage-1 score;
        # add country-relative IDF similarities except the two scale-shifted max-unmatched-token weights (A8)
        sh("exp.py", "--tag", S2, "--train", "normal", "--evals", "normal",
           "--invariant", "size+comp+s1", "--extra", "A8")
        sh("final2.py", "stack", "--s2", S2, "--tag", TAG)
    if stage in ("holdout", "all"):
        sh("final.py", "holdout", "--tag", TAG, "--policy", POLICY)
    if stage in ("fit", "all"):
        sh("final2.py", "fit", "--s2", S2, "--tag", TAG)
    if stage in ("infer", "all"):
        sh("final2.py", "infer", "--tag", TAG, "--policy", POLICY, "--adapt", "--final")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
