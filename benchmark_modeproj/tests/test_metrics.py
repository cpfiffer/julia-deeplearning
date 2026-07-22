"""Unit tests for the environment-independent metric core (numpy only).

These are the scientifically decisive pieces and are runnable without torch /
pyg / the DFT data. Run: python -m pytest benchmark_modeproj/tests -q
or: python benchmark_modeproj/tests/test_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (  # noqa: E402
    evaluation_grid, per_molecule_sis, per_band_sis, ensemble_mean_spectrum,
    paired_delta_sis, BANDS,
)


def _gaussian(grid, center, width, amp=1.0):
    return amp * np.exp(-0.5 * ((grid - center) / width) ** 2)


def test_grid_and_bands():
    g = evaluation_grid()
    assert g[0] == 400.0 and g[-1] <= 4000.0 + 1e-9
    assert set(BANDS) == {"fingerprint", "xh_stretch", "overtone_combination"}


def test_sis_identical_is_one():
    g = evaluation_grid()
    p = _gaussian(g, 1700, 20)[None, :]
    sis = per_molecule_sis(p, p.copy())
    assert np.allclose(sis, 1.0, atol=1e-9), sis


def test_sis_monotone_with_shift():
    g = evaluation_grid()
    truth = _gaussian(g, 1700, 20)[None, :]
    near = _gaussian(g, 1710, 20)[None, :]
    far = _gaussian(g, 2200, 20)[None, :]
    s_near = per_molecule_sis(near, truth)[0]
    s_far = per_molecule_sis(far, truth)[0]
    assert 1.0 > s_near > s_far > 0.0, (s_near, s_far)


def test_ensemble_mean_shape():
    m = np.abs(np.random.default_rng(0).normal(size=(5, 7, 100)))
    mean = ensemble_mean_spectrum(m)
    assert mean.shape == (7, 100)
    assert np.allclose(mean, m.mean(axis=0))


def test_per_band_sis_keys():
    g = evaluation_grid()
    truth = (_gaussian(g, 1000, 15) + _gaussian(g, 3200, 25))[None, :]
    pred = truth.copy()
    bands = per_band_sis(pred, truth, g)
    assert set(bands) == set(BANDS)
    for v in bands.values():
        assert 0.0 < v <= 1.0 + 1e-9


def test_paired_delta_win_rule():
    # candidate strictly better on every pair -> ci_low should exceed 0
    rng = np.random.default_rng(1)
    base = rng.uniform(0.6, 0.8, size=50)
    cand = base + rng.uniform(0.02, 0.05, size=50)  # candidate always higher
    res = paired_delta_sis(cand, base, n_boot=2000, seed=0)
    assert res["delta_sis"] > 0
    assert res["ci_low"] > 0
    assert res["win"] is True
    assert res["n_pairs"] == 50


def test_paired_delta_no_win_when_noise():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.6, 0.8, size=50)
    cand = base + rng.normal(0.0, 0.03, size=50)  # zero-mean difference
    res = paired_delta_sis(cand, base, n_boot=2000, seed=0)
    # not a guaranteed no-win, but CI should straddle 0 for zero-mean noise
    assert res["ci_low"] <= 0 <= res["ci_high"], res


def test_paired_delta_alignment_guard():
    try:
        paired_delta_sis([0.1, 0.2, 0.3], [0.1, 0.2])
    except ValueError:
        return
    raise AssertionError("expected ValueError on misaligned arrays")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\nall {len(fns)} metric tests passed")
