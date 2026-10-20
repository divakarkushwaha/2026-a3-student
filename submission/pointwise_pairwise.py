"""
submission/pointwise_pairwise.py -- the pointwise / pairwise models (assignment Sections 4.1, 4.3 and 7).
Read docs/RANKNET_PRIMER.md first; no prior neural-network experience is assumed.

Both models use the SAME scoring network (harness/neural.py: ScoringNetwork), so the only difference between
them is the loss you write:
  pointwise: squared error between the network's score and the graded label
  pairwise : RankNet -- for a pair (i more relevant than j), loss =
             log(1 + exp(-(s_i - s_j))), i.e. -log sigmoid(s_i - s_j)

PROVIDED : ScoringNetwork, make_scorer, save_scorer, load_scorer (harness/neural.py), dev_ndcg, and the
           overall shape of train_model().
YOU WRITE: every `TODO(...)` below (roughly 100-150 lines in total):
  TODO(1) set_seed            TODO(2) pointwise_loss     TODO(3) make_pairs
  TODO(4) ranknet_loss        TODO(5) pointwise_step / pairwise_step
  TODO(6) the training loop inside train_model(): minibatching, per-epoch dev
          nDCG@10, early stopping.
ALLOWED  : PyTorch tensors, autograd, nn layers, activations, optimizers
           (Adam/SGD), random-seed utilities.
NOT ALLOWED: any ready-made ranking loss or LTR trainer (allRank, TF-Ranking,
           PT-Ranking, RankLib, ...) and LightGBM/XGBoost for these two
           models. torch's MarginRankingLoss is a HINGE loss, not RankNet.

TWO JOBS, KEEP THEM SEPARATE.
  1. Correctness (hand-checked): the functions below are tested by NAME and SIGNATURE (docs/GRADING.md) against
     their textbook definitions. Keep them exactly as declared -- in particular ranknet_loss must stay the plain,
     unweighted RankNet loss.
  2. Track P (leaderboard): your submitted pairwise_model.pt is scored on held-out data. To make it rank better you
     MAY change, in your own extra functions / train_model options: how pairs are sampled (e.g. favour large
     label gaps or top-ranked documents), pairs per query, epochs, learning rate, batch size, the MLP's width and
     depth (at most MAX_HIDDEN_LAYERS hidden layers of at most MAX_WIDTH units: pass e.g. hidden=[128, 64]), and
     per-pair weights such as |delta nDCG| (which turns RankNet into LambdaRank) -- write such weighting as a
     separate function (e.g. weighted_ranknet_loss), never inside ranknet_loss.
     You may NOT: train on anything but relevance labels (no distillation from LambdaMART or any other model's
     scores), or change the artifact format: staff load it with their own copy of harness/neural.py, and an
     artifact that breaks the bounds scores 0 on Track P. Declare the splits you trained and tuned on.

Artifacts to ship in submission/: pointwise_model.pt and pairwise_model.pt (see scripts/train_neural.py).
"""
import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from harness.metrics import mean_ndcg_from_scores
from harness.preprocessing import Standardizer
# PROVIDED, and also what staff use to load your artifacts (assignment Section 7, Track P):
from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer  # noqa: F401


def set_seed(seed: int) -> None:
    """TODO(1): make training reproducible -- seed Python's `random`, NumPy,
    and torch (torch.manual_seed). Staff check two runs with the same seed
    give identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """TODO(2): mean squared error between predicted scores and graded labels
    (both 1-D tensors of equal length). Returns a scalar tensor.
    Example: scores=[1, 2], labels=[0, 2] -> ((1-0)^2 + (2-2)^2) / 2 = 0.5"""
    return ((scores - labels) ** 2).mean()


def make_pairs(qids, labels, pairs_per_query: int, rng: np.random.RandomState) -> np.ndarray:
    """TODO(3): build within-query preference pairs.

    qids  : one query id per ROW (any hashable; rows of a query may be anywhere)
    labels: one graded relevance label per row
    Return an int array of shape (P, 2) of ROW INDICES: column 0 is the
    strictly MORE relevant document of the pair, column 1 the less relevant one.
    Rules (all checked by staff): (a) both rows from the SAME query; (b) only
    pairs with different labels (ties carry no preference); (c) oriented
    more-relevant first; (d) at most `pairs_per_query` pairs per query -- sample
    without replacement using `rng` (a NumPy RandomState) so runs are
    reproducible. Never build the pairs of the whole TRAIN split at once.
    Example: qids=[a,a,a,b,b], labels=[2,0,1,1,1], pairs_per_query large ->
    the 3 pairs (0,1), (0,2), (2,1); query b has only ties -> no pairs."""
    qids = np.asarray(qids)
    labels = np.asarray(labels)
    out = []
    # group row indices by query (rows of a query may be anywhere)
    order = np.argsort(qids, kind="stable")
    sorted_q = qids[order]
    bounds = np.flatnonzero(np.r_[True, sorted_q[1:] != sorted_q[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        lab = labels[idx]
        # all ordered (i, j) with lab[i] > lab[j]: only strictly different grades, more relevant first
        gt = lab[:, None] > lab[None, :]
        ii, jj = np.nonzero(gt)
        n = len(ii)
        if n == 0:
            continue
        if n > pairs_per_query:
            pick = rng.choice(n, size=pairs_per_query, replace=False)   # without replacement
            ii, jj = ii[pick], jj[pick]
        out.append(np.stack([idx[ii], idx[jj]], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out, axis=0).astype(np.int64)


def ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor) -> torch.Tensor:
    """TODO(4): RankNet loss. For each pair, log(1 + exp(-(s_i - s_j))), where
    s_preferred = s_i is the score of the more relevant document. Average over
    the pairs (scalar tensor). Use a numerically stable form -- softplus(-d) or
    torch.logaddexp(0, -d) with d = s_i - s_j -- not a raw exp.
    Example: s_preferred=[2, 0], s_other=[0, 0] ->
    mean(log(1+e^-2), log 2) = mean(0.126928, 0.693147) = 0.410038"""
    return torch.logaddexp(torch.zeros_like(s_preferred), -(s_preferred - s_other)).mean()


def pointwise_step(network: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """TODO(5): ONE optimisation step on a minibatch of documents: zero the
    gradients, forward pass, pointwise_loss, loss.backward(), optimizer.step().
    Return the loss as a float. (Forgetting optimizer.zero_grad() is the classic
    bug: gradients silently accumulate across batches.)"""
    optimizer.zero_grad()
    loss = pointwise_loss(network(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def pairwise_step(network: nn.Module, optimizer, x_preferred: torch.Tensor, x_other: torch.Tensor) -> float:
    """TODO(5): ONE optimisation step on a minibatch of PAIRS. x_preferred /
    x_other are (batch, n_features) feature tensors of each pair's more / less
    relevant document. Run both through the SAME network (shared weights),
    ranknet_loss on the two score vectors, backward, step. Return the loss as a
    float. Staff check one step on a pair raises s_preferred - s_other."""
    optimizer.zero_grad()
    loss = ranknet_loss(network(x_preferred), network(x_other))   # same network => shared weights
    loss.backward()
    optimizer.step()
    return float(loss.item())


def weighted_ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Optional (Track P only, NOT used by the graded ranknet_loss): RankNet with per-pair weights,
    e.g. weights = |gain_i - gain_j| gives a LambdaRank-flavoured loss."""
    per_pair = torch.logaddexp(torch.zeros_like(s_preferred), -(s_preferred - s_other))
    return (weights * per_pair).sum() / weights.sum().clamp_min(1e-12)


# ---- Track P extras (NOT used by the graded functions above) ----------------

def make_pairs_weighted(qids, labels, pairs_per_query: int, rng: np.random.RandomState,
                        gap_power: float = 1.0):
    """Like make_pairs, but when a query has more candidate pairs than pairs_per_query, sample them with
    probability proportional to (gain_i - gain_j)**gap_power, so large-label-gap pairs are favoured. Returns
    (pairs, weights) with weights = |2^l_i - 2^l_j| (LambdaRank-flavoured). Labels only -- no distillation."""
    qids = np.asarray(qids); labels = np.asarray(labels)
    out, wts = [], []
    order = np.argsort(qids, kind="stable"); sq = qids[order]
    bounds = np.flatnonzero(np.r_[True, sq[1:] != sq[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]; lab = labels[idx]
        ii, jj = np.nonzero(lab[:, None] > lab[None, :])
        if len(ii) == 0:
            continue
        gap = np.abs(2.0 ** lab[ii] - 2.0 ** lab[jj])
        if len(ii) > pairs_per_query:
            p = gap ** gap_power; p = p / p.sum()
            pick = rng.choice(len(ii), size=pairs_per_query, replace=False, p=p)
            ii, jj, gap = ii[pick], jj[pick], gap[pick]
        out.append(np.stack([idx[ii], idx[jj]], axis=1)); wts.append(gap)
    if not out:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    return np.concatenate(out).astype(np.int64), np.concatenate(wts).astype(np.float32)


def weighted_pairwise_step(network, optimizer, x_pref, x_other, w) -> float:
    optimizer.zero_grad()
    loss = weighted_ranknet_loss(network(x_pref), network(x_other), w)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---- provided helpers -------------------------------------------------------

def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """PROVIDED. Mean nDCG@10 of `scorer` over the queries of (X, y, qids). It scores
    WHOLE queries and sorts them -- evaluation is not batched like training."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


# ---- training ---------------------------------------------------------------

def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False, weighted: bool = False, weight_decay: float = 0.0):
    """Train a 'pointwise' or 'pairwise' scorer. train / dev are (X_raw, y, qids)
    tuples. Returns (network, standardizer, history) with the best-dev-nDCG
    weights restored. The setup below is PROVIDED; the loop is yours (TODO(6))."""
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    X, y, q = train
    Xd, yd, qd = dev
    set_seed(seed)                                    # TODO(1) makes this meaningful
    standardizer = Standardizer().fit(X)              # fit on TRAIN only
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.RandomState(seed)

    history = []
    best_nd, best_state, bad = -1.0, copy.deepcopy(network.state_dict()), 0
    scorer = make_scorer(network, standardizer)
    for epoch in range(1, epochs + 1):
        network.train()
        if kind == "pairwise":
            if weighted:   # Track P option: gap-favouring sampling + |gain gap| weights
                pairs, pw = make_pairs_weighted(q, y, pairs_per_query, rng)
            else:
                pairs, pw = make_pairs(q, y, pairs_per_query, rng), None
            perm_p = rng.permutation(len(pairs))                   # reshuffle every epoch
            pairs = pairs[perm_p]
            pw = None if pw is None else torch.from_numpy(pw[perm_p])
            n_items = len(pairs)
        else:
            n_items = len(Xt)
            perm = rng.permutation(n_items)
        losses = []
        for start in range(0, n_items, batch_size):
            if kind == "pairwise":
                b = pairs[start:start + batch_size]
                if pw is not None:
                    losses.append(weighted_pairwise_step(network, optimizer, Xt[b[:, 0]], Xt[b[:, 1]],
                                                         pw[start:start + batch_size]))
                else:
                    losses.append(pairwise_step(network, optimizer, Xt[b[:, 0]], Xt[b[:, 1]]))
            else:
                b = perm[start:start + batch_size]
                losses.append(pointwise_step(network, optimizer, Xt[b], yt[b]))
        nd = dev_ndcg(scorer, Xd, yd, qd)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_ndcg@10": float(nd)})
        if verbose:
            print(f"[{kind}] epoch {epoch:2d} loss {np.mean(losses):.4f} dev nDCG@10 {nd:.4f}")
        if nd > best_nd:
            best_nd, best_state, bad = nd, copy.deepcopy(network.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    network.load_state_dict(best_state)
    return network, standardizer, history
