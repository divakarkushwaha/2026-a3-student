"""Reproduce submission/model.txt. TRAIN only by default (declare in README).
    python scripts/train_model.py --train data/train --out submission/model.txt \
        --trees 300 --num-leaves 15 --learning-rate 0.05 --min-data-in-leaf 100
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
from harness.feature_io import read_svmlight_ltr, stack
from scripts.lambdamart_utils import default_params, fit, ndcg10

def _p(path, name): return os.path.join(path, name) if os.path.isdir(path) else path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train"); ap.add_argument("--dev", default="data/dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--trees", type=int, default=300); ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--lambda-l2", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--use-dev", action="store_true")
    a = ap.parse_args()
    tr = read_svmlight_ltr(_p(a.train, "train.txt"))
    dv = read_svmlight_ltr(_p(a.dev, "vali.txt")) if os.path.exists(_p(a.dev, "vali.txt")) else None
    used = {**tr, **dv} if (a.use_dev and dv) else tr
    X, y, q = stack(used)
    m = fit(X, y, q, default_params(a.num_leaves, a.learning_rate, a.min_data_in_leaf, a.lambda_l2, a.seed), a.trees)
    m.save_model(a.out)
    print("train nDCG@10", round(ndcg10(m, X, y, q), 4))
    if dv and not a.use_dev:
        Xd, yd, qd = stack(dv); print("dev   nDCG@10", round(ndcg10(m, Xd, yd, qd), 4))
