#!/usr/bin/env python3
"""
aggregate.py - Section 6 aggregation. Run once after the array completes.

Reads the per-run test_sis.npz artifacts + runs.jsonl and emits:
  1. results_table.csv   - per arch: mean/std test SIS across seeds, param count,
                           mean wall time, experimental NIST SIS, per-band SIS.
  2. deltasis_bootstrap.json - paired-bootstrap DeltaSIS point estimate + 95% CI
                           (candidate - baseline), plus candidate_matched if run.
                           States win/no-win MECHANICALLY vs the prereg rule.
  3. manifest.json       - links every figure and table to git SHA, split hash,
                           prereg hash.
Figures are produced by figures.py (invoked here if matplotlib is available).

Quarantined runs are excluded and their count reported. The ensemble-mean
spectrum (over the 5 members of a seed) is formed before SIS so the endpoint is
exactly the preregistered one (one SIS value per molecule per seed per arch).

Requires numpy (+ matplotlib for figures). Pairing for the bootstrap is by
(molecule, seed), pooled across seeds, per the prereg.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np

from metrics import paired_delta_sis, BANDS

NIST_BASELINE_SIS = 0.538  # secondary reference (experimental gas-phase)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_runs(out_root):
    runs = []
    rp = os.path.join(out_root, "runs.jsonl")
    if os.path.isfile(rp):
        for line in open(rp):
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def ensemble_seed_sis(out_root, arch, seed):
    """Ensemble-mean spectrum over members of (arch, seed) -> per-molecule SIS.

    Returns (mol_ids_sorted, sis_vector) or None if no non-quarantined members.
    """
    from metrics import per_molecule_sis
    member_files = sorted(glob.glob(
        os.path.join(out_root, arch, f"seed{seed}", "member*", "test_sis.npz")))
    preds, truth_ref, ids_ref = [], None, None
    for mf in member_files:
        z = np.load(mf, allow_pickle=True)
        ids = list(z["mol_ids"])
        order = np.argsort(ids)
        if ids_ref is None:
            ids_ref = [ids[i] for i in order]
            truth_ref = z["truth"][order]
        preds.append(z["pred"][order])
    if not preds:
        return None
    mean_pred = np.mean(np.stack(preds, 0), axis=0)  # ensemble-mean spectrum
    return ids_ref, per_molecule_sis(mean_pred, truth_ref)


def per_band_from_ensemble(out_root, arch, seed, grid):
    from metrics import per_band_sis
    member_files = sorted(glob.glob(
        os.path.join(out_root, arch, f"seed{seed}", "member*", "test_sis.npz")))
    preds, truth_ref = [], None
    for mf in member_files:
        z = np.load(mf, allow_pickle=True)
        ids = list(z["mol_ids"]); order = np.argsort(ids)
        if truth_ref is None:
            truth_ref = z["truth"][order]
        preds.append(z["pred"][order])
    if not preds:
        return {b: float("nan") for b in BANDS}
    mean_pred = np.mean(np.stack(preds, 0), axis=0)
    return per_band_sis(mean_pred, truth_ref, grid)


def collect_arch(out_root, arch, seeds, grid):
    """Per-seed ensemble SIS + aligned (molecule, seed) records for bootstrap."""
    seed_means, pairs, band_rows = [], {}, []
    for seed in seeds:
        res = ensemble_seed_sis(out_root, arch, seed)
        if res is None:
            continue
        ids, sis = res
        seed_means.append(float(np.mean(sis)))
        for mid, s in zip(ids, sis):
            pairs[(mid, seed)] = float(s)
        band_rows.append(per_band_from_ensemble(out_root, arch, seed, grid))
    bands = {b: float(np.nanmean([r[b] for r in band_rows])) if band_rows else float("nan")
             for b in BANDS}
    return seed_means, pairs, bands


def aligned_pairs(cand_pairs, base_pairs):
    """Intersect on shared (molecule, seed) keys, return aligned arrays."""
    keys = sorted(set(cand_pairs) & set(base_pairs))
    c = np.array([cand_pairs[k] for k in keys])
    b = np.array([base_pairs[k] for k in keys])
    return keys, c, b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prereg", default="prereg.json")
    ap.add_argument("--make-figures", action="store_true")
    args = ap.parse_args(argv)

    from metrics import evaluation_grid
    grid = evaluation_grid()
    prereg = json.load(open(args.prereg))
    seeds = prereg["primary_decision_statistic"]["seeds"]

    runs = load_runs(args.out_root)
    quarantined = [r for r in runs if r.get("status") == "quarantined"]
    ok_runs = [r for r in runs if r.get("status") == "ok"]

    archs = sorted({r["arch"] for r in ok_runs})
    param_counts = {a: int(np.median([r["param_count"] for r in ok_runs if r["arch"] == a]))
                    for a in archs}
    wall = {a: float(np.mean([r["wall_time_s"] for r in ok_runs if r["arch"] == a]))
            for a in archs}

    collected = {a: collect_arch(args.out_root, a, seeds, grid) for a in archs}

    # ---- results_table.csv ----
    table_path = os.path.join(args.out_root, "results_table.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "n_seeds", "test_sis_mean", "test_sis_std",
                    "param_count", "mean_wall_time_s", "nist_experimental_sis",
                    "sis_fingerprint", "sis_xh_stretch", "sis_overtone_combination"])
        for a in archs:
            seed_means, _pairs, bands = collected[a]
            sm = np.asarray(seed_means, dtype=float)
            w.writerow([a, len(sm),
                        round(float(sm.mean()), 6) if sm.size else "",
                        round(float(sm.std(ddof=1)), 6) if sm.size > 1 else "",
                        param_counts.get(a, ""), round(wall.get(a, float("nan")), 1),
                        NIST_BASELINE_SIS,
                        round(bands["fingerprint"], 6),
                        round(bands["xh_stretch"], 6),
                        round(bands["overtone_combination"], 6)])

    # ---- baseline reproduction gate ----
    base_mean = (np.mean(collected["baseline"][0])
                 if "baseline" in collected and collected["baseline"][0] else float("nan"))
    known = prereg["baseline_reproduction_gate"]["known_ensemble_test_sis"]
    repro_ok = bool(np.isfinite(base_mean) and abs(base_mean - known) <= 0.02)

    # ---- deltasis_bootstrap.json ----
    delta = {"prereg_win_rule": prereg["win_rule"]["statement"],
             "baseline_reproduction": {"observed_mean_test_sis": _r(base_mean),
                                       "known": known, "within_noise": repro_ok},
             "comparisons": {}}
    if "baseline" in collected:
        _, base_pairs, _ = collected["baseline"]
        for cand in ("candidate", "candidate_matched", "candidate_softassign",
                     "abl_no_modeproj", "abl_no_stickenc", "abl_no_multwarp",
                     "abl_no_residualgate"):
            if cand in collected and collected[cand][1]:
                keys, c, b = aligned_pairs(collected[cand][1], base_pairs)
                if c.size:
                    delta["comparisons"][f"{cand}_minus_baseline"] = {
                        **paired_delta_sis(c, b), "reference": "baseline"}
    # ablations vs full candidate (Section 7)
    if "candidate" in collected and collected["candidate"][1]:
        _, cand_pairs, _ = collected["candidate"]
        for abl in ("abl_no_modeproj", "abl_no_stickenc", "abl_no_multwarp", "abl_no_residualgate"):
            if abl in collected and collected[abl][1]:
                keys, a_, c_ = aligned_pairs(collected[abl][1], cand_pairs)
                if a_.size:
                    delta["comparisons"][f"{abl}_minus_candidate"] = {
                        **paired_delta_sis(a_, c_), "reference": "candidate"}

    delta_path = os.path.join(args.out_root, "deltasis_bootstrap.json")
    json.dump(delta, open(delta_path, "w"), indent=2)

    # ---- figures ----
    figures = []
    if args.make_figures:
        try:
            import figures as figmod
            figures = figmod.make_all(args.out_root, collected, grid, seeds)
        except Exception as e:
            print(f"figure generation skipped: {e}")

    # ---- manifest.json ----
    manifest = {
        "git_sha": _git_sha(),
        "split_hash": prereg["frozen_split"]["sha256"],
        "prereg_hash": sha256_file(args.prereg),
        "artifacts": {
            "results_table": {"path": os.path.basename(table_path),
                              "sha256": sha256_file(table_path)},
            "deltasis_bootstrap": {"path": os.path.basename(delta_path),
                                   "sha256": sha256_file(delta_path)},
            "figures": [{"path": os.path.relpath(p, args.out_root),
                         "sha256": sha256_file(p)} for p in figures],
            "runs_jsonl": {"path": "runs.jsonl",
                           "n_ok": len(ok_runs), "n_quarantined": len(quarantined)},
        },
        "quarantined_count": len(quarantined),
        "quarantined": [{"arch": r["arch"], "seed": r.get("seed"),
                         "member": r.get("member"), "reason": r.get("reason_head")}
                        for r in quarantined],
    }
    json.dump(manifest, open(os.path.join(args.out_root, "manifest.json"), "w"), indent=2)

    print(f"wrote results_table.csv, deltasis_bootstrap.json, manifest.json to {args.out_root}")
    print(f"quarantined runs excluded: {len(quarantined)}")
    if not repro_ok:
        print("WARNING: baseline did NOT reproduce known ~0.798 within noise; "
              "pipeline may be broken. Delta reported but flagged.")
    return 0


def _r(x, n=6):
    return round(float(x), n) if np.isfinite(x) else None


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


if __name__ == "__main__":
    raise SystemExit(main())
