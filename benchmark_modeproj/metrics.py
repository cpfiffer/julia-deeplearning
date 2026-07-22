"""
metrics.py - evaluation metrics for the mode-projection delta-ML benchmark.

The SIS / SID / paired_bootstrap_delta primitives are imported unchanged from
the vendored spectral-active-learning skill kernel (spectral_active_learning.py)
so the benchmark uses byte-for-byte the same metric and CI machinery the
preregistration mandates. This module only adds:

  * a fixed evaluation grid definition and the three spectral bands
    (fingerprint / X-H stretch / overtone-combination),
  * per-band SIS,
  * the ensemble-mean spectrum protocol,
  * paired-bootstrap DeltaSIS pooled over (molecule, seed) pairs, expressed
    directly on top of the skill's paired_bootstrap_delta so the CI routine is
    identical to the one used elsewhere in the project.

No metric is redefined here; everything decisive routes through the vendored
kernel. Nothing in this module fabricates or synthesises spectra.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from spectral_active_learning import (
    spectral_information_similarity,
    spectral_information_divergence,
    paired_bootstrap_delta,
)

# --------------------------------------------------------------------------- #
# evaluation grid + bands
# --------------------------------------------------------------------------- #
# Shared evaluation grid for BOTH arms. Matched-comparison control #4: the grid
# and the SIS implementation are identical across arms. These are the standard
# gas-phase mid-IR limits used by the Paper 1 pipeline; override via prereg if
# the frozen pipeline uses a different grid (assert_grid_matches_prereg).
GRID_MIN_CM = 400.0
GRID_MAX_CM = 4000.0
GRID_STEP_CM = 2.0


def evaluation_grid(vmin: float = GRID_MIN_CM,
                    vmax: float = GRID_MAX_CM,
                    step: float = GRID_STEP_CM) -> np.ndarray:
    """The cm^-1 grid on which both arms' spectra are broadened and compared."""
    return np.arange(vmin, vmax + 0.5 * step, step, dtype=np.float64)


# Band edges in cm^-1. Reported (not decisive) per the prereg secondary
# endpoints: per-band SIS decomposition.
BANDS = {
    "fingerprint": (400.0, 1500.0),
    "xh_stretch": (2700.0, 3700.0),
    "overtone_combination": (1500.0, 2700.0),
}


def band_mask(grid: np.ndarray, band: str) -> np.ndarray:
    lo, hi = BANDS[band]
    return (grid >= lo) & (grid < hi)


# --------------------------------------------------------------------------- #
# ensemble protocol
# --------------------------------------------------------------------------- #
def ensemble_mean_spectrum(member_spectra: np.ndarray) -> np.ndarray:
    """Mean predicted spectrum over ensemble members (the inference protocol).

    member_spectra : (M, n_mol, F) per-member predicted spectra.
    returns        : (n_mol, F) ensemble-mean spectrum.

    Matched-comparison control #4: identical ensemble protocol for both arms
    (5 independently seeded members, mean spectrum at inference).
    """
    member_spectra = np.asarray(member_spectra, dtype=np.float64)
    if member_spectra.ndim != 3:
        raise ValueError(f"member_spectra must be (M,n_mol,F); got {member_spectra.shape}")
    return member_spectra.mean(axis=0)


# --------------------------------------------------------------------------- #
# per-molecule SIS (primary endpoint building block)
# --------------------------------------------------------------------------- #
def per_molecule_sis(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-molecule SIS between predicted and HF-truth spectra. (n_mol,).

    pred, truth : (n_mol, F). Uses the vendored kernel SIS unchanged.
    """
    return spectral_information_similarity(np.asarray(pred), np.asarray(truth))


def per_band_sis(pred: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> dict:
    """Per-band mean SIS. Secondary endpoint (reported, not decisive).

    Returns {band_name: mean_SIS_over_molecules}. SIS is computed on the band
    slice of the (already broadened, non-negative) spectra; the kernel
    L1-normalises within SIS, so each band is renormalised to its own mass,
    which is the intended within-band comparison.
    """
    out = {}
    for band in BANDS:
        m = band_mask(grid, band)
        out[band] = float(per_molecule_sis(pred[:, m], truth[:, m]).mean())
    return out


# --------------------------------------------------------------------------- #
# primary decision statistic: paired-bootstrap DeltaSIS pooled over (mol, seed)
# --------------------------------------------------------------------------- #
def paired_delta_sis(sis_candidate: Sequence[float],
                     sis_baseline: Sequence[float],
                     n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Paired-bootstrap DeltaSIS = mean(SIS_candidate - SIS_baseline).

    sis_candidate, sis_baseline : equal-length, index-aligned arrays. Each
    element is one (molecule, seed) pair's per-molecule test SIS; pairing is by
    (molecule, seed) as the prereg requires, pooled across the 5 seeds. The two
    arrays MUST be aligned so that element j refers to the same (molecule, seed)
    in both arms.

    Routes the paired resampling through the vendored kernel's
    paired_bootstrap_delta (identical CI machinery to the rest of the project).

    returns dict: point estimate, ci_low, ci_high, n_pairs, win (ci_low > 0).
    """
    a = np.asarray(sis_candidate, dtype=np.float64)
    b = np.asarray(sis_baseline, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"candidate/baseline SIS arrays must align; got {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("no paired (molecule, seed) observations supplied")
    point, ci_low, ci_high = paired_bootstrap_delta(a, b, n_boot=n_boot, seed=seed)
    return {
        "delta_sis": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": 0.95,
        "n_pairs": int(a.size),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "win": bool(ci_low > 0.0),  # prereg win rule: CI lower bound on DeltaSIS > 0
    }
