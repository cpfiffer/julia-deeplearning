"""
losses.py - the composite training loss (Section 3).

  L = lambda1 * pointwise(log-spectrum)
    + lambda2 * Sinkhorn-EMD
    + lambda3 * stick-level (only if mode assignment between fidelities is
      available via displacement-vector overlap; else lambda3 = 0)

The EMD term is essential so a peak shifted a few cm^-1 is not penalised as a
missing peak. Lambdas are tuned on val and FROZEN in prereg before test eval.

Requires torch (env `pyg`). Not executed in the authoring container.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def pointwise_log_loss(pred, truth, eps: float = 1e-8):
    """MSE in log-spectrum space on L1-normalised spectra."""
    p = pred.clamp_min(0)
    t = truth.clamp_min(0)
    p = p / (p.sum(-1, keepdim=True) + eps)
    t = t / (t.sum(-1, keepdim=True) + eps)
    return F.mse_loss(torch.log(p + eps), torch.log(t + eps))


def sinkhorn_emd(pred, truth, grid, eps: float = 1.0, iters: int = 50):
    """Entropic-regularised 1D Wasserstein (Sinkhorn) between spectra as
    distributions over the cm^-1 grid. Penalises transport distance, so a
    shifted peak costs its shift, not its full mass.

    pred, truth : (B, F) non-negative. grid : (F,) cm^-1.
    """
    B, Fdim = pred.shape
    a = pred.clamp_min(1e-8); a = a / a.sum(-1, keepdim=True)
    b = truth.clamp_min(1e-8); b = b / b.sum(-1, keepdim=True)
    # cost = squared distance on the grid, scaled to O(1)
    g = grid.view(1, -1)
    C = (g.view(-1, 1) - g.view(1, -1)) ** 2
    C = C / C.max()
    K = torch.exp(-C / eps)  # (F, F)
    u = torch.ones_like(a)
    for _ in range(iters):
        Kv = b / (torch.einsum("ij,bj->bi", K.t(), u).clamp_min(1e-8))
        u = a / (torch.einsum("ij,bj->bi", K, Kv).clamp_min(1e-8))
    Kv = b / (torch.einsum("ij,bj->bi", K.t(), u).clamp_min(1e-8))
    # transport plan P = diag(u) K diag(Kv); cost = <P, C>
    cost = torch.einsum("bi,ij,bj,ij->b", u, K, Kv, C)
    return cost.mean()


def stick_level_loss(omega_pred, log_i_pred, omega_hf, log_i_hf, assignment_mask):
    """Direct stick-level loss when a fidelity mode assignment exists (via
    displacement-vector overlap). assignment_mask : (B, K) 1 where a HF partner
    is assigned. Zero contribution (and lambda3=0) when unavailable."""
    m = assignment_mask.to(omega_pred.dtype)
    denom = m.sum().clamp_min(1.0)
    freq = (((omega_pred - omega_hf) ** 2) * m).sum() / denom
    inten = (((log_i_pred - log_i_hf) ** 2) * m).sum() / denom
    return freq / (100.0 ** 2) + inten  # scale freq residual to O(1)


def composite_loss(out, batch, grid, lambdas: dict):
    """Assemble L. `lambdas` keys: l1, l2, l3 (l3 applied only if the batch
    carries a stick assignment)."""
    pred = out["spectrum"]
    truth = batch["hf_spectrum"]
    l1 = pointwise_log_loss(pred, truth)
    l2 = sinkhorn_emd(pred, truth, grid,
                      eps=lambdas.get("sinkhorn_eps", 1.0),
                      iters=lambdas.get("sinkhorn_iters", 50))
    total = lambdas["l1"] * l1 + lambdas["l2"] * l2
    parts = {"pointwise": float(l1.detach()), "emd": float(l2.detach())}
    if lambdas.get("l3", 0.0) > 0.0 and batch.get("stick_assignment") is not None:
        l3 = stick_level_loss(out["omega_pred"], out["log_i_pred"],
                              batch["omega_hf"], batch["log_i_hf"],
                              batch["stick_assignment"])
        total = total + lambdas["l3"] * l3
        parts["stick"] = float(l3.detach())
    return total, parts
