# Mode-Projection Delta-ML vs Paper 1 Baseline - campaign code

This directory is the **code deliverable** for the preregistered benchmark that
compares a mode-projection delta-ML spectrum architecture (the CANDIDATE) against
the Paper 1 dual-encoder architecture (the BASELINE) under matched controls.

## Status: code authored here; NO training runs and NO result artifacts were produced in this session

Read this first. It is the honest scope of what this session could and could not do.

The campaign brief targets **UNSW Katana** (PBS Pro, conda env `pyg`, scratch
`/srv/scratch/z5076150/`) with the **frozen Paper 1 pipeline, split, LF/HF DFT
data, and broadening module** already present. This session ran instead in an
ephemeral cloud container attached to the `just-cameron/julia-deeplearning`
repository (a Julia/fast.ai course website). In this container there is:

- no Katana / PBS scheduler,
- no conda `pyg` environment (bare Python 3.11; `torch`/`torch_geometric` absent),
- no `/srv/scratch/z5076150/` and no archived DFT data (`.hess`, LF/HF pairs, sticks),
- no frozen split file, and no Paper 1 model to reproduce the known ~0.798 SIS.

The primary endpoint is a preregistered, paired-bootstrap significance test over
**50 trained models** on frozen DFT data. Those models **cannot be trained here**.
The one outcome worse than not running the benchmark would be emitting
`results_table.csv` / `deltasis_bootstrap.json` with **fabricated numbers**, which
would poison the separate interpretation session that treats these artifacts as
frozen ground truth. So this session stopped at the gate and produced **code to
run on Katana**, not results.

**To actually run the campaign:** move this directory to Katana (or run from the
Paper 1 repo with this package on `PYTHONPATH`), set the `TODO_PIPELINE` hooks to
the frozen pipeline's real modules/paths, and follow `orchestrate/launch.md`.

## What IS done and verified in this session

- `metrics.py` and the vendored `spectral_active_learning.py` (the SIS/SID and
  `paired_bootstrap_delta` machinery the prereg mandates) - **unit-tested with
  numpy** (`tests/test_metrics.py`, 8 tests pass), including the win-rule logic
  (CI lower bound on ΔSIS > 0).
- `preregister.py` - the Section 0 gate - **tested** (`tests/test_prereg.py`):
  refuses to emit when the frozen split is missing; records the win rule, seeds,
  split hash, and starts `loss_lambdas.frozen = false` so a downgrade cannot pass
  silently.
- Identity-init correction heads (`models/heads.py`) carry a torch self-test
  (`python models/heads.py` under `pyg`) asserting HF = LF at init.

## What is written to spec but NOT executed (needs `pyg` + frozen pipeline)

`broadening.py` (shared pseudo-Voigt), `models/candidate.py` (two encoders,
mode-projection readout, gated cross-fusion, decomposed heads), `models/baseline.py`
(thin wrapper importing the unchanged Paper 1 model), `losses.py` (pointwise-log +
Sinkhorn-EMD + optional stick-level), `data.py` (split-hash assertion + Section 2
dependency-check / fallback recorder), `train.py` (one arch/seed/member, early
stopping, `runs.jsonl` append, quarantine-on-failure), `aggregate.py` (results
table, paired-bootstrap ΔSIS, manifest), `figures.py`, and `orchestrate/`.

## Layout

```
preregister.py            Section 0 gate -> prereg.json (tested)
metrics.py                grid, bands, ensemble mean, per-molecule/per-band SIS,
                          paired-bootstrap ΔSIS + win rule (tested)
spectral_active_learning.py   vendored skill kernel (SIS/SID, paired_bootstrap_delta)
config.py                 SharedConfig - matched controls for BOTH arms
broadening.py             SharedPseudoVoigt - the ONE broadening layer both arms use
models/heads.py           identity-init frequency / intensity / gated-residual heads
models/candidate.py       CANDIDATE (mode-projection) + ablation variants
models/baseline.py        BASELINE (unchanged Paper 1) wrapper
losses.py                 composite loss (pointwise-log + Sinkhorn-EMD + stick)
data.py                   split-hash assert + Section 2 dependency check/fallback
train.py                  one (arch, seed, member); runs.jsonl; quarantine
aggregate.py              results_table.csv, deltasis_bootstrap.json, manifest.json
figures.py                scatter, ΔSIS distribution, representative spectra
orchestrate/array_job.pbs 50-task PBS array (2 arms x 5 seeds x 5 members)
orchestrate/smoke.sh      budget-first smoke task
orchestrate/launch.md     end-to-end launch order
tests/                    numpy/stdlib unit tests (runnable anywhere)
```

## `TODO_PIPELINE` hooks to wire before launch

- `config.SHARED.split_file`, `config.SHARED.out_root`, DimeNet++ hyperparameters,
  LR schedule / batch-size constants - set to the frozen Paper 1 values.
- `models/baseline.py`: `PAPER1_MODULE` / `PAPER1_FACTORY` (the published model factory).
- `train.py`: `paper1.data.build_dataloaders`, `paper1.models.dimenetpp.build_geometry_encoder`.
- `data._archive_has`: the LF archive layout for the dependency check.
- `prereg.loss_lambdas`: tune on val, then set `frozen = true`.

## No fabricated science

Nothing in this package invents spectra, SIS values, or ΔSIS results. All decisive
numbers come from the vendored metric kernel run on real model outputs, which do
not exist until the array runs on Katana.
