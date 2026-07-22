#!/usr/bin/env python3
"""
preregister.py - Section 0 preregistration gate.

Emits prereg.json and REFUSES to proceed if the frozen split hash is missing.
Run and commit this before any model code trains. The win rule is recorded here,
before any result exists.

Usage:
    python preregister.py --split-file /path/to/frozen_split.json \
                          --out prereg.json

The split file must already exist and carry a stable content hash. This script
does not create, resample, or modify the split; it only reads and freezes its
hash into the preregistration. If the split file is absent, the gate fails hard
(non-zero exit) so no arm can be trained against an unfrozen split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(default: str = "UNKNOWN") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return default


def build_prereg(split_file: str, split_hash: str) -> dict:
    """Assemble the preregistration record. Pure; no side effects."""
    return {
        "campaign": "mode-projection delta-ML vs Paper 1 baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "frozen_split": {
            "file": os.path.abspath(split_file),
            "sha256": split_hash,
        },
        # ------------------------------------------------------------------ #
        # Primary endpoint
        # ------------------------------------------------------------------ #
        "primary_endpoint": {
            "name": "per-molecule test-set SIS on the frozen held-out test split",
            "spectrum": "ensemble-mean predicted spectrum (5 members)",
            "aggregation": "averaged over molecules; one value per architecture per seed",
        },
        "primary_decision_statistic": {
            "name": "paired-bootstrap DeltaSIS",
            "definition": "mean(SIS_candidate - SIS_baseline) over shared test molecules, "
                          "pooled across the 5 seeds; pair by (molecule, seed)",
            "ci": "95% paired-bootstrap CI via spectral_active_learning.paired_bootstrap_delta",
            "seeds": [0, 1, 2, 3, 4],
            "members_per_seed": 5,
        },
        # ------------------------------------------------------------------ #
        # Win rule - recorded now, before any result exists
        # ------------------------------------------------------------------ #
        "win_rule": {
            "statement": "CANDIDATE wins iff the 95% paired-bootstrap CI lower "
                         "bound on DeltaSIS > 0.",
            "decisive": True,
            "mechanical": "report ci_low vs 0; win == ci_low > 0",
        },
        # ------------------------------------------------------------------ #
        # Secondary endpoints - reported, not decisive
        # ------------------------------------------------------------------ #
        "secondary_endpoints": {
            "experimental_nist_gasphase_sis": {"baseline_reference": 0.538},
            "per_band_sis": ["fingerprint", "xh_stretch", "overtone_combination"],
            "param_count_per_arch": True,
        },
        # ------------------------------------------------------------------ #
        # Matched-comparison controls (Section 1)
        # ------------------------------------------------------------------ #
        "matched_controls": {
            "shared_split": True,
            "shared_lf_hf_pairs": "PBEh-3c -> wB97X-D3/def2-TZVPD",
            "shared_broadening_module": "single learnable pseudo-Voigt instance for both arms",
            "shared_eval_grid_and_sis": "imported from spectral-active-learning kernel",
            "shared_ensemble_protocol": "5 seeded members, mean spectrum at inference",
            "shared_training_protocol": {
                "optimizer": "AdamW",
                "lr_schedule": "matched",
                "max_epochs": "matched",
                "early_stopping": "val SIS, matched patience",
                "batch_size": "matched",
                "augmentation": "matched",
            },
            "param_match_tolerance": "candidate within +/-20% of baseline; report both; "
                                     "if larger, also run candidate_matched",
            "seeds": [0, 1, 2, 3, 4],
        },
        # ------------------------------------------------------------------ #
        # Loss lambdas - placeholder; MUST be tuned on val and frozen here
        # BEFORE any test evaluation. Left null so the gate cannot be passed
        # off as final until the tuning step writes them back.
        # ------------------------------------------------------------------ #
        "loss_lambdas": {
            "lambda1_pointwise_log": None,
            "lambda2_sinkhorn_emd": None,
            "lambda3_stick_level": None,
            "frozen": False,
            "note": "tune on val split, then freeze (set frozen=true) before test eval",
        },
        # ------------------------------------------------------------------ #
        # Candidate testability flags (Section 2 decision defaults) - filled by
        # the data dependency check; recorded here so a downgrade is never silent.
        # ------------------------------------------------------------------ #
        "candidate_testability": {
            "displacement_vectors_available": None,
            "per_mode_lf_sticks_available": None,
            "true_mode_projection_arm_testable": None,
            "fallback_arm_tag": None,
            "notes": "set by data.check_dependencies(); do not train until resolved",
        },
        "evaluation_grid": {
            "vmin_cm": 400.0, "vmax_cm": 4000.0, "step_cm": 2.0,
            "note": "assert against the frozen pipeline grid before test eval",
        },
        "baseline_reproduction_gate": {
            "known_ensemble_test_sis": 0.798,
            "rule": "if baseline does not reproduce within noise, STOP and flag "
                    "the pipeline as broken rather than reporting a delta",
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preregistration gate for the mode-projection benchmark.")
    p.add_argument("--split-file", required=True,
                   help="Path to the existing frozen train/val/test split file.")
    p.add_argument("--out", default="prereg.json", help="Output preregistration path.")
    args = p.parse_args(argv)

    if not os.path.isfile(args.split_file):
        sys.stderr.write(
            f"PREREG GATE FAILED: frozen split file not found: {args.split_file}\n"
            "Refusing to proceed. The split must exist and be frozen before "
            "preregistration.\n"
        )
        return 2

    split_hash = sha256_file(args.split_file)
    if not split_hash:
        sys.stderr.write("PREREG GATE FAILED: could not hash split file.\n")
        return 2

    prereg = build_prereg(args.split_file, split_hash)
    with open(args.out, "w") as f:
        json.dump(prereg, f, indent=2, sort_keys=False)

    prereg_hash = sha256_file(args.out)
    sys.stdout.write(
        f"Preregistration written to {args.out}\n"
        f"  frozen split sha256: {split_hash}\n"
        f"  prereg sha256:       {prereg_hash}\n"
        "Commit prereg.json before training. Downstream scripts must assert the "
        "split hash matches this record.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
