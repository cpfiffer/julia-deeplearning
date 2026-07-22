"""
config.py - shared, matched training/eval configuration for BOTH arms.

Section 1 matched-comparison controls live here so the two arms cannot drift.
Both baseline and candidate import this same object. Anything arch-specific
(width, depth) is set per-arch but the shared protocol below is identical.

NOTE: values marked TODO_PIPELINE must be reconciled against the frozen Paper 1
pipeline before launch (e.g. exact DimeNet++ hyperparameters, exact LR schedule
constants). They are set to the Paper 1 defaults named in the campaign brief;
confirm them against the frozen codebase and record any change in prereg.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class SharedConfig:
    # --- data / split ---
    split_file: str = "/srv/scratch/z5076150/paper1/frozen_split.json"  # TODO_PIPELINE
    lf_level: str = "PBEh-3c"
    hf_level: str = "wB97X-D3/def2-TZVPD"

    # --- evaluation grid (must match metrics.evaluation_grid & prereg) ---
    grid_vmin_cm: float = 400.0
    grid_vmax_cm: float = 4000.0
    grid_step_cm: float = 2.0

    # --- ensemble ---
    seeds: tuple = (0, 1, 2, 3, 4)
    members_per_seed: int = 5

    # --- training protocol (matched across arms) ---
    optimizer: str = "AdamW"
    lr: float = 3e-4                 # TODO_PIPELINE confirm vs Paper 1
    weight_decay: float = 1e-5       # TODO_PIPELINE
    lr_schedule: str = "cosine"      # TODO_PIPELINE
    warmup_epochs: int = 5
    max_epochs: int = 300
    batch_size: int = 32             # TODO_PIPELINE
    early_stop_metric: str = "val_sis"
    early_stop_mode: str = "max"
    early_stop_patience: int = 30
    grad_clip: float = 5.0
    augmentation: str = "none"       # TODO_PIPELINE (match whatever Paper 1 uses)

    # --- loss lambdas: tuned on val, then FROZEN in prereg before test eval ---
    lambda1_pointwise_log: float = 1.0
    lambda2_sinkhorn_emd: float = 0.1
    lambda3_stick_level: float = 0.0     # 0 unless displacement-overlap assignment exists
    sinkhorn_eps: float = 1.0
    sinkhorn_iters: int = 50

    # --- shared model width (used to param-match the arms) ---
    hidden_dim: int = 128

    # --- output / bookkeeping ---
    out_root: str = "/srv/scratch/z5076150/benchmark_modeproj"
    runs_jsonl: str = "runs.jsonl"

    def to_dict(self) -> dict:
        return asdict(self)


SHARED = SharedConfig()


# DimeNet++ hyperparameters reused verbatim for BOTH geometry encoders (parity).
# TODO_PIPELINE: replace with the exact frozen Paper 1 DimeNet++ config.
DIMENETPP = dict(
    hidden_channels=128,
    out_channels=128,
    num_blocks=4,
    int_emb_size=64,
    basis_emb_size=8,
    out_emb_channels=256,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0,
    max_num_neighbors=32,
    envelope_exponent=5,
    num_before_skip=1,
    num_after_skip=2,
    num_output_layers=3,
)
