"""
baseline.py - the current Paper 1 architecture (Section 4), unchanged.

DimeNet++ + Conv-ViT dual encoder; warp / global / fine-residual gated heads;
learnable pseudo-Voigt; 5-member ensemble. Trained on the identical split with
the identical training protocol and the SAME broadening module instance as the
candidate.

This wrapper is intentionally thin: the campaign brief says to train the Paper 1
architecture UNCHANGED. The authoritative implementation lives in the frozen
Paper 1 codebase; import it here rather than re-implementing so the baseline is
byte-for-byte the published model. The only shared object injected from this
benchmark is the SharedPseudoVoigt broadening instance (matched-comparison
control #3).

If the frozen Paper 1 module is importable, `build_baseline` returns it wired to
the shared broadening layer. If it is not on PYTHONPATH, `build_baseline` raises
with an explicit message rather than silently substituting a re-implementation
(that would break the "reproduces known ensemble test SIS ~0.798" gate).

Requires torch (env `pyg`). Not executed in the authoring container.
"""
from __future__ import annotations

import importlib

# Dotted path to the frozen Paper 1 model factory. TODO_PIPELINE: set to the
# real module (e.g. "paper1.models.dual_encoder"). Kept as a name so the
# benchmark does not fork the published architecture.
PAPER1_MODULE = "paper1.models.dual_encoder"
PAPER1_FACTORY = "build_dual_encoder_delta_model"


def build_baseline(broadening, dimenetpp_cfg: dict, hidden_dim: int, seed: int):
    """Instantiate the unchanged Paper 1 model wired to the SHARED broadening.

    Parameters
    ----------
    broadening : the SharedPseudoVoigt instance shared with the candidate arm.
    dimenetpp_cfg : the frozen DimeNet++ hyperparameters (config.DIMENETPP).
    hidden_dim : shared width.
    seed : member seed (weight init only; data order handled by the trainer).
    """
    try:
        mod = importlib.import_module(PAPER1_MODULE)
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"Could not import the frozen Paper 1 model '{PAPER1_MODULE}'. "
            "The baseline arm MUST be the unchanged published architecture, not "
            "a re-implementation. Put the Paper 1 package on PYTHONPATH (env "
            "`pyg`) and set PAPER1_MODULE/PAPER1_FACTORY in baseline.py. "
            f"Underlying error: {e}"
        )
    factory = getattr(mod, PAPER1_FACTORY)
    # The Paper 1 factory is expected to accept an external broadening module so
    # both arms share the SAME instance. If the published factory constructs its
    # own broadening, adapt here (and record the adaptation in prereg notes).
    return factory(broadening=broadening, dimenetpp_cfg=dimenetpp_cfg,
                   hidden_dim=hidden_dim, seed=seed)
