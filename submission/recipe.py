"""submission/recipe.py -- Track C1 transfer recipe. Dimension/label agnostic, deterministic, time-boxed.
Hyperparameters are RULES of data size/width, then the tree count is chosen on the benchmark's own vali file
(early stopping is legitimate here: vali is given to us for exactly that). Final model = best iteration."""
import time
import numpy as np
from harness.feature_io import read_svmlight_ltr, stack


def _group_sizes(qids):
    sizes, last = [], object()
    for q in qids:
        if q == last: sizes[-1] += 1
        else: sizes.append(1); last = q
    return sizes


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    import lightgbm as lgb
    t0 = time.time()
    X, y, q = stack(read_svmlight_ltr(train_path))
    Xv, yv, qv = stack(read_svmlight_ltr(vali_path, n_features=X.shape[1]))
    n_rows, n_feat = X.shape
    n_q = len(set(q.tolist()))
    # rules, not MSLR constants
    num_leaves = int(np.clip(round(np.sqrt(n_rows) / 8), 7, 31))
    min_leaf = int(np.clip(n_rows // 1500, 20, 200))
    ff = float(np.clip(40.0 / max(n_feat, 1) + 0.5, 0.5, 0.9))
    max_label = int(max(y.max(), yv.max()))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "label_gain": [2 ** i - 1 for i in range(max(max_label, 4) + 1)],
        "num_leaves": num_leaves, "learning_rate": 0.05, "min_data_in_leaf": min_leaf,
        "lambda_l2": 5.0, "feature_fraction": ff, "bagging_fraction": 0.8, "bagging_freq": 1,
        "max_bin": 127, "seed": seed, "deterministic": True, "force_row_wise": True,
        "verbose": -1, "num_threads": 4,
    }
    # time budget: ~12 of the 20 minutes; cap trees by a cheap per-tree timing estimate
    max_trees = 600 if n_rows <= 200_000 else 350
    dtr = lgb.Dataset(X, label=y, group=_group_sizes(q))
    dva = lgb.Dataset(Xv, label=yv, group=_group_sizes(qv), reference=dtr)

    class _Budget:
        def __call__(self, env):
            if time.time() - t0 > 11 * 60:
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    booster = lgb.train(params, dtr, num_boost_round=max_trees, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), _Budget()])
    booster.save_model(out_model_path, num_iteration=booster.best_iteration or -1)
