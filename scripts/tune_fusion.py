"""Tune fusion (method, RRF c, channel weights) on TRAIN only -> submission/fusion_config.json.
    python scripts/tune_fusion.py --train data/train/train.txt
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from harness.feature_io import read_svmlight_ltr
from harness.feature_names import BASE_RANKER_CHANNELS
from harness.metrics import ndcg_at_k
from submission.fusion import rrf_scores, combsum_scores, combmnz_scores

def evaluate(data, fn, max_q=None):
    vals = []
    for i, qd in enumerate(data.values()):
        if max_q and i >= max_q: break
        bs = {d: {c: float(qd.features[r, idx - 1]) for c, idx in BASE_RANKER_CHANNELS.items()}
              for r, d in enumerate(qd.doc_ids)}
        fused = fn(bs)
        ranked = sorted(bs, key=lambda d: -fused[d])
        qrels = {d: int(l) for d, l in zip(qd.doc_ids, qd.labels)}
        vals.append(ndcg_at_k(ranked, qrels, 10))
    return float(np.mean(vals))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train/train.txt")
    ap.add_argument("--max-q", type=int, default=800)
    ap.add_argument("--out", default="submission/fusion_config.json")
    a = ap.parse_args()
    data = read_svmlight_ltr(a.train)
    chans = list(BASE_RANKER_CHANNELS)
    grid_w = [0.0, 1.0, 2.0]
    best = (-1, None)
    for method in ("rrf", "combsum", "combmnz"):
        cs = (20, 60) if method == "rrf" else (None,)
        for c in cs:
            for w in itertools.product(grid_w, repeat=len(chans)):
                if sum(w) == 0: continue
                wd = dict(zip(chans, w))
                fn = {"rrf": lambda b: rrf_scores(b, c=c, weights=wd),
                      "combsum": lambda b: combsum_scores(b, weights=wd),
                      "combmnz": lambda b: combmnz_scores(b, weights=wd)}[method]
                s = evaluate(data, fn, a.max_q)
                if s > best[0]:
                    best = (s, {"method": method, "c": c, "weights": wd}); print(f"{s:.4f}", best[1])
    json.dump(best[1], open(a.out, "w"), indent=2)
    print("TRAIN nDCG@10 of best fusion:", round(best[0], 4), "->", a.out)
