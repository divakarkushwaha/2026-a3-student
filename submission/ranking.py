"""submission/ranking.py -- competition entry. Fusion config (from scripts/tune_fusion.py, TRAIN only)
is read from fusion_config.json if present, else plain RRF c=60."""
import json, os
from typing import Dict, List, Tuple
import numpy as np
from submission.fusion import rrf_scores, combsum_scores, combmnz_scores

_MODEL = None
_here = os.path.dirname(os.path.abspath(__file__))
try:
    _CFG = json.load(open(os.path.join(_here, "fusion_config.json")))
except Exception:
    _CFG = {"method": "rrf", "c": 60, "weights": None}


def prepare(model_path: str) -> None:
    global _MODEL
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"trained model artifact not found at {model_path!r}")
    import lightgbm as lgb
    _MODEL = lgb.Booster(model_file=model_path)


def _unique(ids): return list(dict.fromkeys(ids))


def fuse_candidates(qid: str, candidate_doc_ids: List[str], base_scores: Dict[str, Dict[str, float]],
                    k: int) -> List[Tuple[str, float]]:
    ids = _unique(candidate_doc_ids)
    bs = {d: base_scores[d] for d in ids}
    m, w = _CFG.get("method", "rrf"), _CFG.get("weights")
    if m == "combsum": fused = combsum_scores(bs, weights=w)
    elif m == "combmnz": fused = combmnz_scores(bs, weights=w)
    else: fused = rrf_scores(bs, c=_CFG.get("c") or 60.0, weights=w)
    ranked = sorted(ids, key=lambda d: -fused[d])
    return [(d, float(fused[d])) for d in ranked[:k]]


def rerank_with_ltr(qid: str, candidate_doc_ids: List[str], features: Dict[str, List[float]],
                    k: int) -> List[Tuple[str, float]]:
    if _MODEL is None: raise RuntimeError("call prepare(model_path) first")
    ids = _unique(candidate_doc_ids)
    if not ids: return []
    X = np.asarray([features[d] for d in ids], dtype=np.float64)
    preds = _MODEL.predict(X)
    order = np.argsort(-preds, kind="stable")
    return [(ids[i], float(preds[i])) for i in order[:k]]
