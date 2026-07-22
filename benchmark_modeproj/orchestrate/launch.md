# Launch order (Katana, env `pyg`)

Run from `benchmark_modeproj/` with the frozen Paper 1 package on `PYTHONPATH`.

1. **Preregister (gate).** Requires the frozen split to exist.
   ```
   python preregister.py --split-file /srv/scratch/z5076150/paper1/frozen_split.json \
                         --out /srv/scratch/z5076150/benchmark_modeproj/prereg.json
   git add prereg.json && git commit -m "preregister mode-projection benchmark"
   ```

2. **Dependency check + record the candidate variant** (Section 2). Writes the
   `candidate_testability` block back into `prereg.json`; a downgrade is recorded,
   never silent.
   ```
   python -c "import json,data; \
     t=data.check_dependencies('/srv/scratch/z5076150/paper1/lf_archive'); \
     data.merge_testability_into_prereg('/srv/scratch/z5076150/benchmark_modeproj/prereg.json', t); \
     print(json.dumps(t, indent=2))"
   ```

3. **Tune loss lambdas on val, then FREEZE** them in `prereg.json`
   (`loss_lambdas.frozen = true`). Training refuses to start until frozen.

4. **Smoke task** (budget-first). Must pass before the array.
   ```
   bash orchestrate/smoke.sh candidate 0 0
   ```

5. **Full head-to-head array** - 50 tasks (2 arms x 5 seeds x 5 members).
   ```
   qsub -J 0-49 orchestrate/array_job.pbs
   ```
   Baseline reproduction gate: confirm the baseline ensemble test SIS reproduces
   the known ~0.798 within noise (aggregate.py flags this). If not, STOP.

6. **Aggregate + figures** after the array drains.
   ```
   python aggregate.py --out-root /srv/scratch/z5076150/benchmark_modeproj --make-figures
   ```
   Emits `results_table.csv`, `deltasis_bootstrap.json`, `figures/`, `manifest.json`.

7. **Second wave (gated).** ONLY if the primary DeltaSIS 95% CI excludes 0.
   Launch the four ablation arms (`abl_no_modeproj`, `abl_no_stickenc`,
   `abl_no_multwarp`, `abl_no_residualgate`), each 5 seeds x 5 members, same
   controls, with an index map analogous to `array_job.pbs`. Re-run `aggregate.py`.

## Param-count matching (Section 1.5)
After a candidate build, compare its param count to the baseline. If the
candidate exceeds baseline by > 20%, set `candidate_matched_width` in `prereg.json`
to the width that brings it within tolerance and run the `candidate_matched` arm
(same 5x5). Both counts are reported in `results_table.csv`.
