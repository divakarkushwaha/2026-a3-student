"""Grouped 5-fold CV sweep (no early stopping). Saves cv_results.csv for your report plots.
    python scripts/cv_lambdamart.py --train data/train --trees 50 100 200 400 --num-leaves 7 15 31
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse, csv
import numpy as np
from harness.cv import grouped_kfold
from harness.feature_io import read_svmlight_ltr, stack
from harness.metrics import mean_ndcg_from_scores
from scripts.lambdamart_utils import default_params, fit, ndcg10

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train"); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--num-leaves", type=int, nargs="+", default=[15])
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--lambda-l2", type=float, default=5.0); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="cv_results.csv")
    a = ap.parse_args()
    path = os.path.join(a.train, "train.txt") if os.path.isdir(a.train) else a.train
    X, y, q = stack(read_svmlight_ltr(path))
    folds = grouped_kfold(q, k=a.folds, seed=a.seed)
    rows = []
    print(f"{'leaves':>6}{'trees':>7}{'train':>9}{'val':>9}{'val_std':>9}{'val/train':>10}")
    for nl in a.num_leaves:
        p = default_params(nl, a.learning_rate, a.min_data_in_leaf, a.lambda_l2, a.seed, num_threads=a.threads)
        per = {n: ([], []) for n in a.trees}
        for ti, vi in folds:   # train ONCE per fold at max trees, score at each fixed tree count (no early stopping)
            m = fit(X[ti], y[ti], q[ti], p, max(a.trees))
            for n in a.trees:
                per[n][0].append(mean_ndcg_from_scores(m.predict(X[ti], num_iteration=n), y[ti], q[ti], k=10))
                per[n][1].append(mean_ndcg_from_scores(m.predict(X[vi], num_iteration=n), y[vi], q[vi], k=10))
        for n in a.trees:
            tr, va = per[n]
            r = (nl, n, np.mean(tr), np.mean(va), np.std(va), np.mean(va) / np.mean(tr)); rows.append(r)
            print(f"{nl:>6}{n:>7}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>9.4f}{r[5]:>10.4f}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["num_leaves", "trees", "train", "val_mean", "val_std", "ratio"]); w.writerows(rows)