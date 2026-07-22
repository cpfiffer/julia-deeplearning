"""
data.py - data interface + Section 2 dependency check / decision defaults.

Reuses the existing Paper 1 data pipeline. This module does NOT define new data;
it (a) loads the frozen split and asserts its hash against prereg, and (b) runs
the dependency check that decides which CANDIDATE variant is testable:

  * displacement vectors L (N_modes, N_atoms, 3) + per-mode LF (omega_k, I_k)
    archived alongside the LF calcs  -> true mode-projection readout.
  * displacement vectors NOT archived -> attempt cheap re-extraction from ORCA
    .hess files; if infeasible in budget -> FALL BACK to learned soft-assignment
    attention pooling readout, tag `candidate_softassign`, record in prereg that
    the true mode-projection arm was not testable. NEVER silently substitute.
  * per-mode LF sticks exist -> encode directly. Only broadened LF trace exists
    -> fall back to 1D Conv/ViT over the trace and note the downgrade.

The heavy lifting (ORCA .hess parsing, dataset assembly) is delegated to the
frozen Paper 1 pipeline; this file only orchestrates the decision and records it.

Requires numpy; torch only for the Dataset/collate used at train time.
"""
from __future__ import annotations

import hashlib
import json
import os


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split_and_assert(split_file: str, prereg_path: str) -> dict:
    """Load the frozen split and assert its hash matches the preregistration.

    Matched-comparison control #1 + Section 0 gate. Raises on any mismatch so no
    arm can train against a split that differs from the one preregistered.
    """
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"frozen split not found: {split_file}")
    if not os.path.isfile(prereg_path):
        raise FileNotFoundError(f"prereg.json not found: {prereg_path}; run preregister.py first")
    prereg = json.load(open(prereg_path))
    recorded = prereg["frozen_split"]["sha256"]
    actual = sha256_file(split_file)
    if actual != recorded:
        raise ValueError(
            "SPLIT HASH MISMATCH: the split file does not match the "
            f"preregistration.\n  prereg: {recorded}\n  actual: {actual}\n"
            "Refusing to proceed; the comparison would not be the preregistered one."
        )
    split = json.load(open(split_file))
    for key in ("train", "val", "test"):
        if key not in split:
            raise ValueError(f"split file missing '{key}' partition")
    return split


def check_dependencies(lf_archive_dir: str) -> dict:
    """Decide which CANDIDATE variant is testable (Section 2). Returns a record
    to merge into prereg.json under 'candidate_testability'.

    This inspects the LF archive for displacement vectors and per-mode sticks.
    The concrete checks are delegated to the frozen pipeline's archive layout;
    the fields below are what the decision must record. A downgrade is always
    written back to prereg, never applied silently.
    """
    have_disp = _archive_has(lf_archive_dir, "displacement_vectors")
    have_sticks = _archive_has(lf_archive_dir, "per_mode_sticks")
    have_hess = _archive_has(lf_archive_dir, "orca_hess")
    have_trace = _archive_has(lf_archive_dir, "broadened_lf_trace")

    # readout decision
    if have_disp:
        readout, arm_tag, true_testable = "modeproj", "candidate", True
        reextract = False
    elif have_hess:
        # attempt cheap re-extraction from .hess; feasibility flag set by the
        # extractor. If it succeeds, we still get the true mode-projection arm.
        readout, arm_tag, true_testable = "modeproj", "candidate", True
        reextract = True
    else:
        readout, arm_tag, true_testable = "softassign", "candidate_softassign", False
        reextract = False

    # input-encoding decision
    if have_sticks:
        input_encoding, input_downgrade = "stick_set_transformer", False
    elif have_trace:
        input_encoding, input_downgrade = "trace_conv_vit", True
    else:
        raise RuntimeError(
            "Neither per-mode LF sticks nor a broadened LF trace found in "
            f"{lf_archive_dir}; cannot build the LF-spectrum encoder."
        )

    return {
        "displacement_vectors_available": bool(have_disp),
        "per_mode_lf_sticks_available": bool(have_sticks),
        "orca_hess_available": bool(have_hess),
        "attempt_hess_reextraction": bool(reextract),
        "true_mode_projection_arm_testable": bool(true_testable),
        "readout_variant": readout,
        "fallback_arm_tag": None if arm_tag == "candidate" else arm_tag,
        "input_encoding": input_encoding,
        "input_encoding_downgraded_to_trace": bool(input_downgrade),
        "notes": ("true mode-projection arm testable" if true_testable else
                  "displacement vectors unavailable and .hess re-extraction "
                  "infeasible; using learned soft-assignment attention pooling "
                  "(candidate_softassign). Recorded, not silently substituted."),
    }


def merge_testability_into_prereg(prereg_path: str, testability: dict) -> None:
    """Write the dependency-check outcome back into prereg.json (append-style
    update of the candidate_testability block; other fields untouched)."""
    prereg = json.load(open(prereg_path))
    prereg["candidate_testability"] = {**prereg.get("candidate_testability", {}), **testability}
    with open(prereg_path, "w") as f:
        json.dump(prereg, f, indent=2, sort_keys=False)


def _archive_has(lf_archive_dir: str, what: str) -> bool:
    """Probe the LF archive for a data kind. TODO_PIPELINE: implement against
    the frozen archive layout. Returns False conservatively if the archive is
    absent so the caller falls back / records a downgrade rather than assuming.
    """
    if not lf_archive_dir or not os.path.isdir(lf_archive_dir):
        return False
    markers = {
        "displacement_vectors": ("displacements.npy", "modes_L.npz", "L.npy"),
        "per_mode_sticks": ("sticks.npz", "lf_sticks.json", "modes.csv"),
        "orca_hess": (".hess",),
        "broadened_lf_trace": ("lf_trace.npy", "lf_broadened.npz"),
    }[what]
    for _root, _dirs, files in os.walk(lf_archive_dir):
        for fn in files:
            if any(fn.endswith(m) or fn == m for m in markers):
                return True
    return False
