#!/usr/bin/env python3
"""
train.py - train ONE (arch, seed, member) model end to end.

One PBS array index -> one call to this script. On success it:
  * writes a checkpoint under out_root/<arch>/seed<seed>/member<member>/,
  * appends one line to runs.jsonl (append-only registry, Section 0),
  * writes the per-molecule test SIS vector needed by aggregation.
On non-convergence / failure it QUARANTINES the run (moves partial artifacts to
out_root/quarantine/ with a failure reason) and appends a quarantine line;
quarantined runs are excluded from the aggregate (Section 5).

Both arms share config.SHARED (matched-comparison controls). The candidate arm
uses the readout/encoding variant chosen by data.check_dependencies and recorded
in prereg. The baseline arm is the unchanged Paper 1 model.

Requires torch + the frozen Paper 1 pipeline (env `pyg`). Not executed in the
authoring container. The control flow, artifact contract, and registry schema
are the deliverable; the numeric training runs on Katana.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import time
import traceback

# torch imported lazily inside main() so --help works without the env.


ARCHS = ("baseline", "candidate", "candidate_matched",
         "candidate_softassign",
         # second-wave ablations (Section 7), launched only if primary signals:
         "abl_no_modeproj", "abl_no_stickenc", "abl_no_multwarp", "abl_no_residualgate")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train one (arch, seed, member).")
    p.add_argument("--arch", required=True, choices=ARCHS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--member", type=int, required=True)
    p.add_argument("--prereg", default="prereg.json")
    p.add_argument("--out-root", default=None, help="override config.SHARED.out_root")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (few epochs, few molecules) for the budget-first smoke task")
    return p.parse_args(argv)


def run_dir(out_root, arch, seed, member):
    return os.path.join(out_root, arch, f"seed{seed}", f"member{member}")


def append_jsonl(path, record):
    """Append-only registry write. Never overwrites (Section 0)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def quarantine(out_root, arch, seed, member, reason):
    src = run_dir(out_root, arch, seed, member)
    qdir = os.path.join(out_root, "quarantine", f"{arch}_seed{seed}_member{member}")
    os.makedirs(qdir, exist_ok=True)
    if os.path.isdir(src):
        for fn in os.listdir(src):
            shutil.move(os.path.join(src, fn), os.path.join(qdir, fn))
    with open(os.path.join(qdir, "FAILURE_REASON.txt"), "w") as f:
        f.write(reason + "\n")
    return qdir


def git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


def build_model(arch, broadening, prereg, seed):
    """Wire the requested arch. Imports the frozen Paper 1 pieces lazily."""
    from config import DIMENETPP, SHARED
    d = SHARED.hidden_dim
    if arch == "baseline":
        from models.baseline import build_baseline
        return build_baseline(broadening, DIMENETPP, d, seed)

    # geometry encoder is the frozen Paper 1 DimeNet++ for BOTH arms (parity)
    from models.baseline import build_baseline  # reused to fetch the geom encoder
    geom = _frozen_geometry_encoder(DIMENETPP, d, seed)

    from models.candidate import ModeProjectionDeltaML
    tb = prereg.get("candidate_testability", {})
    use_sticks = tb.get("input_encoding", "stick_set_transformer") == "stick_set_transformer"
    readout = "modeproj"
    freq_form, residual_gated = "mult", True

    if arch == "candidate_softassign":
        readout = "modeproj"  # soft-assign readout selected inside model when disp absent
    if arch == "abl_no_modeproj":
        readout = "global"
    if arch == "abl_no_stickenc":
        use_sticks = False
    if arch == "abl_no_multwarp":
        freq_form = "add"
    if arch == "abl_no_residualgate":
        residual_gated = False

    width = d if arch != "candidate_matched" else _matched_width(prereg, d)
    return ModeProjectionDeltaML(geom, broadening, d=width,
                                 readout_variant=readout, freq_form=freq_form,
                                 residual_gated=residual_gated,
                                 use_stick_encoder=use_sticks)


def _frozen_geometry_encoder(cfg, d, seed):
    import importlib
    mod = importlib.import_module("paper1.models.dimenetpp")  # TODO_PIPELINE
    return mod.build_geometry_encoder(cfg, hidden_dim=d, seed=seed)


def _matched_width(prereg, d):
    # width-reduced candidate matched to baseline params (Section 1.5). The
    # concrete width is computed by the param-count reconciliation step and
    # recorded in prereg; fall back to d if not yet set.
    return int(prereg.get("candidate_matched_width", d))


def main(argv=None):
    args = parse_args(argv)

    import numpy as np
    import torch

    from config import SHARED
    from broadening import SharedPseudoVoigt
    from metrics import evaluation_grid, per_molecule_sis, per_band_sis
    from losses import composite_loss
    from data import load_split_and_assert
    # frozen pipeline dataset assembly:
    from paper1.data import build_dataloaders  # TODO_PIPELINE

    out_root = args.out_root or SHARED.out_root
    prereg = json.load(open(args.prereg))
    if not prereg.get("loss_lambdas", {}).get("frozen", False):
        raise SystemExit("prereg loss_lambdas not frozen; tune on val and freeze before training.")

    torch.manual_seed(args.member)  # member controls weight init -> ensemble diversity
    np.random.seed(args.member)

    rd = run_dir(out_root, args.arch, args.seed, args.member)
    os.makedirs(rd, exist_ok=True)
    t0 = time.time()

    try:
        split = load_split_and_assert(SHARED.split_file, args.prereg)
        grid = torch.as_tensor(evaluation_grid(), dtype=torch.float32)
        broadening = SharedPseudoVoigt(grid)  # shared-semantics instance
        model = build_model(args.arch, broadening, prereg, args.seed)

        train_loader, val_loader, test_loader = build_dataloaders(
            split, SHARED, seed=args.seed, smoke=args.smoke)

        lam = prereg["loss_lambdas"]
        lambdas = {"l1": lam["lambda1_pointwise_log"], "l2": lam["lambda2_sinkhorn_emd"],
                   "l3": lam["lambda3_stick_level"],
                   "sinkhorn_eps": SHARED.sinkhorn_eps, "sinkhorn_iters": SHARED.sinkhorn_iters}

        opt = torch.optim.AdamW(model.parameters(), lr=SHARED.lr,
                                weight_decay=SHARED.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SHARED.max_epochs)

        best_val, best_state, patience = -1.0, None, 0
        max_epochs = 3 if args.smoke else SHARED.max_epochs
        param_count = sum(p.numel() for p in model.parameters())

        for epoch in range(max_epochs):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                out = model(batch)
                loss, _ = composite_loss(out, batch, grid, lambdas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), SHARED.grad_clip)
                opt.step()
            sched.step()

            val_sis = _eval_sis(model, val_loader)
            if val_sis > best_val + 1e-5:
                best_val, best_state, patience = val_sis, _cpu_state(model), 0
            else:
                patience += 1
                if patience >= SHARED.early_stop_patience:
                    break

        if best_state is None or best_val <= 0.0:
            raise RuntimeError(f"non-convergence: best_val_sis={best_val}")

        model.load_state_dict(best_state)
        test_pred, test_truth, mol_ids = _predict(model, test_loader)
        sis_vec = per_molecule_sis(test_pred, test_truth)
        bands = per_band_sis(test_pred, test_truth, evaluation_grid())

        torch.save({"state_dict": best_state, "param_count": param_count},
                   os.path.join(rd, "checkpoint.pt"))
        np.savez(os.path.join(rd, "test_sis.npz"),
                 mol_ids=np.asarray(mol_ids), sis=sis_vec,
                 pred=test_pred, truth=test_truth)

        wall = time.time() - t0
        record = {
            "arch": args.arch, "seed": args.seed, "member": args.member,
            "git_sha": git_sha(), "split_hash": prereg["frozen_split"]["sha256"],
            "prereg_hash": _sha256(args.prereg),
            "param_count": int(param_count), "wall_time_s": round(wall, 1),
            "final_val_sis": round(float(best_val), 6),
            "test_sis_mean": round(float(sis_vec.mean()), 6),
            "per_band_sis": bands, "host": socket.gethostname(),
            "status": "ok", "smoke": bool(args.smoke),
        }
        append_jsonl(os.path.join(out_root, SHARED.runs_jsonl), record)
        print(json.dumps(record))
        return 0

    except Exception:
        reason = traceback.format_exc()
        qdir = quarantine(out_root, args.arch, args.seed, args.member, reason)
        append_jsonl(os.path.join(out_root, SHARED.runs_jsonl), {
            "arch": args.arch, "seed": args.seed, "member": args.member,
            "git_sha": git_sha(), "status": "quarantined",
            "quarantine_dir": qdir, "wall_time_s": round(time.time() - t0, 1),
            "reason_head": reason.strip().splitlines()[-1][:200],
        })
        print(f"QUARANTINED -> {qdir}")
        return 1


# --- small helpers (torch present) ---------------------------------------- #
def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _eval_sis(model, loader):
    import numpy as np
    from metrics import per_molecule_sis
    pred, truth, _ = _predict(model, loader)
    return float(np.mean(per_molecule_sis(pred, truth)))


def _predict(model, loader):
    import numpy as np
    import torch
    model.eval()
    preds, truths, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            preds.append(out["spectrum"].cpu().numpy())
            truths.append(batch["hf_spectrum"].cpu().numpy())
            ids.extend(batch["mol_id"])
    return np.concatenate(preds), np.concatenate(truths), ids


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
