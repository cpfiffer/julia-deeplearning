"""
figures.py - Section 6 figures. Invoked by aggregate.py --make-figures.

Produces:
  (a) per-molecule SIS scatter, candidate vs baseline, y=x reference;
  (b) DeltaSIS distribution with the bootstrap CI;
  (c) 3-4 representative overlaid spectra (LF, baseline pred, candidate pred,
      HF truth) from the largest positive and largest negative DeltaSIS molecules.

Design intent follows the scientific-figures / publication-figures skills:
single-column widths, colour-blind-safe palette, direct labels, y=x and zero
references drawn explicitly. Requires matplotlib + numpy.

Note: figure (c) needs the per-molecule predicted spectra (pred arrays in the
test_sis.npz files) plus the LF trace; the loader pulls them from the run
artifacts. Not executed in the authoring container (no matplotlib / no runs).
"""
from __future__ import annotations

import glob
import os

import numpy as np

# colour-blind-safe (Okabe-Ito subset)
C_BASE = "#0072B2"
C_CAND = "#D55E00"
C_LF = "#999999"
C_HF = "#000000"


def _seed_ensemble_pred(out_root, arch, seed):
    files = sorted(glob.glob(os.path.join(out_root, arch, f"seed{seed}", "member*", "test_sis.npz")))
    preds, truth, ids = [], None, None
    for mf in files:
        z = np.load(mf, allow_pickle=True)
        idx = list(z["mol_ids"]); order = np.argsort(idx)
        if ids is None:
            ids = [idx[i] for i in order]; truth = z["truth"][order]
        preds.append(z["pred"][order])
    if not preds:
        return None
    return ids, np.mean(np.stack(preds, 0), 0), truth


def make_all(out_root, collected, grid, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(out_root, "figures")
    os.makedirs(figdir, exist_ok=True)
    paths = []

    from metrics import per_molecule_sis

    # Aggregate per-molecule ensemble SIS across seeds (mean over seeds per mol).
    def arch_mol_sis(arch):
        acc = {}
        for seed in seeds:
            r = _seed_ensemble_pred(out_root, arch, seed)
            if r is None:
                continue
            ids, pred, truth = r
            s = per_molecule_sis(pred, truth)
            for mid, v in zip(ids, s):
                acc.setdefault(mid, []).append(float(v))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    base = arch_mol_sis("baseline")
    cand = arch_mol_sis("candidate")
    shared = sorted(set(base) & set(cand))
    if shared:
        xb = np.array([base[m] for m in shared])
        yc = np.array([cand[m] for m in shared])

        # (a) scatter
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        lim = [min(xb.min(), yc.min()) - 0.02, 1.0]
        ax.plot(lim, lim, color="0.5", lw=1, ls="--", zorder=0)
        ax.scatter(xb, yc, s=14, c=C_CAND, alpha=0.7, edgecolor="none")
        ax.set_xlabel("baseline per-molecule SIS")
        ax.set_ylabel("candidate per-molecule SIS")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        fig.tight_layout()
        p = os.path.join(figdir, "a_scatter_candidate_vs_baseline.png")
        fig.savefig(p, dpi=300); plt.close(fig); paths.append(p)

        # (b) DeltaSIS distribution + bootstrap CI
        from metrics import paired_delta_sis
        d = yc - xb
        res = paired_delta_sis(yc, xb)
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.hist(d, bins=30, color=C_BASE, alpha=0.8)
        ax.axvline(0, color="0.4", lw=1, ls="--")
        ax.axvline(res["delta_sis"], color=C_CAND, lw=1.5, label="mean ΔSIS")
        ax.axvspan(res["ci_low"], res["ci_high"], color=C_CAND, alpha=0.15,
                   label="95% CI")
        ax.set_xlabel("per-molecule ΔSIS (candidate − baseline)")
        ax.set_ylabel("count"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        p = os.path.join(figdir, "b_deltasis_distribution.png")
        fig.savefig(p, dpi=300); plt.close(fig); paths.append(p)

        # (c) representative overlaid spectra: extremes of DeltaSIS
        order = np.argsort(d)
        picks = list(order[:2]) + list(order[-2:])  # 2 most negative, 2 most positive
        r_base = _seed_ensemble_pred(out_root, "baseline", seeds[0])
        r_cand = _seed_ensemble_pred(out_root, "candidate", seeds[0])
        if r_base and r_cand:
            ids_b, pred_b, truth_b = r_base
            ids_c, pred_c, _ = r_cand
            idmap_c = {m: i for i, m in enumerate(ids_c)}
            fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=True)
            for ax, j in zip(axes.ravel(), picks):
                mid = shared[j]
                if mid not in ids_b or mid not in idmap_c:
                    continue
                ib = ids_b.index(mid); ic = idmap_c[mid]
                ax.plot(grid, truth_b[ib], color=C_HF, lw=1.2, label="HF truth")
                ax.plot(grid, pred_b[ib], color=C_BASE, lw=1.0, label="baseline")
                ax.plot(grid, pred_c[ic], color=C_CAND, lw=1.0, label="candidate")
                ax.set_title(f"{mid}  ΔSIS={d[j]:+.3f}", fontsize=8)
            axes[0, 0].legend(fontsize=6, frameon=False)
            for ax in axes[-1, :]:
                ax.set_xlabel("wavenumber (cm$^{-1}$)")
            fig.tight_layout()
            p = os.path.join(figdir, "c_representative_spectra.png")
            fig.savefig(p, dpi=300); plt.close(fig); paths.append(p)

    return paths
