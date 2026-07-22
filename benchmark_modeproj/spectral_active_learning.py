"""
spectral_active_learning — recommended acquisition functions for active
learning of molecular IR spectra with a deep-ensemble corrector.

Recommendation (verified on the homogeneous Seager biosignature pool, Round-1
protocol: 7 rounds, batch 200, 5-member ensemble, 5 seeds, paired-bootstrap
95% CI on area-under-the-learning-curve of SIS):

    C3 — query-by-committee spectral-SID disagreement — is the acquisition of
    choice. On the homogeneous pool it has the highest AULC-OOD (0.6881), ahead
    of random (0.6763) and the top-k-uncertainty baseline B0 (0.6652). Its edge
    over B0 is +0.023 [+0.014, +0.034] (paired-bootstrap 95% CI); its edge over
    random is smaller (+0.012, point estimate) but it is the top-ranked method
    on both in-distribution and out-of-distribution SIS. Pointwise top-k
    uncertainty (B0) performs *below* random on this pool (random beats B0 by
    +0.011 [+0.005, +0.017]) and must be avoided.

Why classic uncertainty sampling fails on homogeneous pools
-----------------------------------------------------------
When the unlabelled pool is chemically self-similar (Seager: 77% of molecules
at exactly 6 heavy atoms, nearest-neighbour Tanimoto ~0.49, feature space
compressed to ~19 effective dimensions of 236), every high-uncertainty
molecule has near-duplicate neighbours the model is equally unsure about.
Top-k selection then loads the batch with redundant twins, so the labelled set
grows in information much slower than its size — and the strategy degenerates
toward random. C3 helps because committee *disagreement* in spectral-divergence
space is less duplicated across near-twins than raw predictive variance.

This module is self-contained (numpy + optional scikit-learn) and framework
agnostic: you supply per-member predicted spectra and (optionally) an
embedding; it returns the indices to label next.

Author: generated for R. McAlister's IR-spectrum AL project, 2026-07.
"""
from itertools import combinations
import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# spectral divergence (SID) and the SIS similarity it induces
# --------------------------------------------------------------------------- #
def l1_normalize(spec: np.ndarray) -> np.ndarray:
    """L1-normalise non-negative broadened spectra to probability vectors."""
    spec = np.clip(np.asarray(spec, dtype=np.float64), 0.0, None)
    return spec / (spec.sum(axis=-1, keepdims=True) + EPS)


def spectral_information_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Symmetric spectral information divergence (SID) between rows of p and q.

    p, q : (n, F) non-negative broadened IR spectra. Returns (n,).
    SID(p,q) = sum_nu [ p ln(p/q) + q ln(q/p) ] on L1-normalised spectra.
    """
    p = np.clip(l1_normalize(p), EPS, None)
    q = np.clip(l1_normalize(q), EPS, None)
    return np.sum(p * np.log(p / q) + q * np.log(q / p), axis=-1)


def spectral_information_similarity(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """SIS = 1 / (1 + SID) in [0,1]; 1.0 == identical spectra. (n,)."""
    return 1.0 / (1.0 + spectral_information_divergence(p, q))


# --------------------------------------------------------------------------- #
# uncertainty / disagreement scalars from an ensemble's per-member spectra
# --------------------------------------------------------------------------- #
def total_spectral_variance(member_preds: np.ndarray) -> np.ndarray:
    """B0 score: sum over bins of across-member variance. (M,n,F) -> (n,).

    Provided for completeness / as the documented *negative* control — this is
    the pointwise-uncertainty signal that collapses to random on homogeneous
    pools. Do not use it as the primary acquisition on such pools.
    """
    return np.sum(np.var(member_preds, axis=0), axis=-1)


def committee_sid_disagreement(member_preds: np.ndarray) -> np.ndarray:
    """C3 score: mean pairwise SID between committee members. (M,n,F) -> (n,).

    The recommended acquisition signal. Higher == members disagree more about
    the spectrum == more informative to label. Robust on homogeneous pools
    because disagreement is less duplicated across near-twin molecules than
    raw predictive variance.
    """
    member_preds = np.asarray(member_preds, dtype=np.float64)
    if member_preds.ndim != 3:
        raise ValueError(f"member_preds must be (M,n,F); got {member_preds.shape}")
    M = member_preds.shape[0]
    if M < 2:
        raise ValueError("committee disagreement needs >= 2 ensemble members")
    pair_vals = [spectral_information_divergence(member_preds[i], member_preds[j])
                 for i, j in combinations(range(M), 2)]
    return np.mean(pair_vals, axis=0)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def minmax_scale(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo <= EPS else (x - lo) / (hi - lo)


def min_sqdist_to_set(emb, ref):
    a2 = np.sum(emb * emb, axis=1, keepdims=True)
    b2 = np.sum(ref * ref, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * (emb @ ref.T), 0.0).min(axis=1)


def select_c3(member_preds: np.ndarray, k: int) -> np.ndarray:
    """RECOMMENDED. Top-k by committee SID disagreement.

    member_preds : (M, n_pool, F) per-member predicted spectra for the pool.
    k            : batch size.
    returns      : (k,) local indices into the pool (highest disagreement first).
    """
    scores = committee_sid_disagreement(member_preds)
    k = min(int(k), len(scores))
    return np.argsort(scores, kind="stable")[-k:][::-1]


def select_c3_diverse(member_preds: np.ndarray, embeddings: np.ndarray, k: int,
                      beta: float = 1.0) -> np.ndarray:
    """Optional anti-redundancy variant (N1): greedy facility-location that
    maximises committee disagreement weighted by distance to the already-
    chosen batch. Use when batch redundancy is a concern on a very homogeneous
    pool; falls back to select_c3's ranking when embeddings are uninformative.

    member_preds : (M, n_pool, F). embeddings : (n_pool, d). returns (k,) local.
    """
    dis = committee_sid_disagreement(member_preds)
    z = np.asarray(embeddings, dtype=np.float64)
    n = len(dis); k = min(int(k), n)
    u = minmax_scale(dis) + 1e-3
    first = int(np.argmax(dis)); chosen = [first]
    min_d = min_sqdist_to_set(z, z[first:first + 1])
    for _ in range(1, k):
        score = u * np.power(np.maximum(min_d, 0.0), beta)
        score[chosen] = -np.inf
        pick = int(np.argmax(score)); chosen.append(pick)
        min_d = np.minimum(min_d, min_sqdist_to_set(z, z[pick:pick + 1]))
    return np.asarray(chosen, dtype=np.int64)


# convenience registry (function, not a module-level dict, so this file also
# loads cleanly as a skill kernel.py sidecar)
def get_acquisition(name):
    """Return an acquisition function by name: 'c3' (recommended) or
    'c3_diverse' (optional anti-redundancy variant)."""
    table = {"c3": select_c3, "c3_diverse": select_c3_diverse}
    if name not in table:
        raise KeyError(f"unknown acquisition {name!r}; known: {sorted(table)}")
    return table[name]


# --------------------------------------------------------------------------- #
# evaluation: area under the learning curve (AULC)
# --------------------------------------------------------------------------- #
def aulc(n_labeled, metric_values):
    """Normalised area under the learning curve: trapezoid(metric vs n) / span.

    n_labeled, metric_values : 1-D arrays of equal length (SIS per round).
    Higher AULC == the metric rose faster per labelled sample. Use as the
    primary sample-efficiency endpoint when ranking acquisition functions.
    """
    n = np.asarray(n_labeled, dtype=np.float64)
    y = np.asarray(metric_values, dtype=np.float64)
    order = np.argsort(n); n, y = n[order], y[order]
    if n[-1] - n[0] <= EPS:
        return float("nan")
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(y, n) / (n[-1] - n[0]))


def paired_bootstrap_delta(aulc_a, aulc_b, n_boot=10000, seed=0):
    """Paired bootstrap of mean(AULC_a - AULC_b) over seeds.

    aulc_a, aulc_b : per-seed AULC arrays (same seeds, same order).
    returns (mean_delta, ci_low, ci_high) at 95%. delta>0 & ci_low>0 == a
    beats b significantly. This is the exact test used to certify C3 over the
    B0 top-k-uncertainty baseline on the Seager pool (+0.023 [+0.014, +0.034]);
    the C3-over-random margin is smaller (+0.012 point estimate) and its CI was
    not separately certified.
    """
    a = np.asarray(aulc_a, dtype=np.float64)
    b = np.asarray(aulc_b, dtype=np.float64)
    d = a - b
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
