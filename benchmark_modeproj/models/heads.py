"""
heads.py - decomposed correction heads (Section 3), all initialised to the
IDENTITY correction so the untrained model predicts HF = LF exactly and training
learns only the delta.

  * FrequencyHead  : omega_pred = omega_LF * (1 + 0.15 * tanh(MLP(m_k))),
                     bounded to [0.85, 1.15] * omega_LF, final layer zero-init
                     so the initial scale is exactly 1.0.
  * IntensityHead  : log I_pred = log I_LF + MLP(m_k), final layer zero-init ->
                     identity.
  * ResidualHead   : additive broad spectrum residual, gated by
                     g = sigmoid(scalar) with the scalar init strongly negative
                     so g ~ 0 at start; residual switches on only if it earns loss.

Requires torch. Not executed in the authoring container. A self-test at the
bottom asserts the identity property when run under `pyg`.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

    class _NNStub:  # import-safety stub; real nn used only under torch (Katana)
        Module = object
    nn = _NNStub()  # type: ignore


def _mlp(d_in, d_hidden, d_out, zero_init_last=True):
    layers = [nn.Linear(d_in, d_hidden), nn.SiLU(),
              nn.Linear(d_hidden, d_hidden), nn.SiLU(),
              nn.Linear(d_hidden, d_out)]
    if zero_init_last:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class FrequencyHead(nn.Module):
    """Per-mode bounded multiplicative frequency warp. Identity at init."""

    def __init__(self, d, hidden=None, max_frac: float = 0.15):
        super().__init__()
        self.max_frac = max_frac
        self.mlp = _mlp(d, hidden or d, 1, zero_init_last=True)

    def forward(self, m, omega_lf):
        # m: (B, K, d); omega_lf: (B, K)
        s = self.max_frac * torch.tanh(self.mlp(m).squeeze(-1))  # zero at init
        scale = 1.0 + s                                          # exactly 1.0 at init
        scale = scale.clamp(1.0 - self.max_frac, 1.0 + self.max_frac)
        return omega_lf * scale


class IntensityHead(nn.Module):
    """Per-mode additive log-intensity correction. Identity at init."""

    def __init__(self, d, hidden=None):
        super().__init__()
        self.mlp = _mlp(d, hidden or d, 1, zero_init_last=True)

    def forward(self, m, log_i_lf):
        return log_i_lf + self.mlp(m).squeeze(-1)  # +0 at init


class ResidualHead(nn.Module):
    """Gated additive spectrum-space residual for features LF misses entirely
    (overtones, combination bands, Fermi resonances). Gate ~ 0 at init."""

    def __init__(self, d, grid_size, hidden=256, gate_init: float = -6.0):
        super().__init__()
        self.pool = nn.Sequential(nn.Linear(d, hidden), nn.SiLU())
        self.decode = nn.Linear(hidden, grid_size)
        nn.init.zeros_(self.decode.weight)
        nn.init.zeros_(self.decode.bias)
        # strongly negative scalar gate -> sigmoid ~ 0.0025 at init
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, mode_tokens, mask=None):
        # mode_tokens: (B, K, d) -> molecule embedding via masked mean
        if mask is not None:
            w = mask.to(mode_tokens.dtype).unsqueeze(-1)
            pooled = (mode_tokens * w).sum(1) / w.sum(1).clamp_min(1e-6)
        else:
            pooled = mode_tokens.mean(1)
        residual = self.decode(self.pool(pooled))  # (B, F); zero at init
        g = torch.sigmoid(self.gate_logit)
        return g * residual, g


def _self_test():  # pragma: no cover - runs only under torch
    torch.manual_seed(0)
    B, K, d, F = 3, 12, 32, 100
    m = torch.randn(B, K, d)
    omega_lf = torch.rand(B, K) * 3000 + 400
    log_i_lf = torch.randn(B, K)
    mask = torch.ones(B, K)

    fh, ih, rh = FrequencyHead(d), IntensityHead(d), ResidualHead(d, F)
    fh.eval(); ih.eval(); rh.eval()
    with torch.no_grad():
        assert torch.allclose(fh(m, omega_lf), omega_lf, atol=1e-5), "freq head not identity at init"
        assert torch.allclose(ih(m, log_i_lf), log_i_lf, atol=1e-5), "intensity head not identity at init"
        res, g = rh(m, mask)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-5), "residual not zero at init"
        assert float(g) < 0.01, f"gate not ~0 at init: {float(g)}"
    print("heads identity self-test passed")


if __name__ == "__main__":
    if not _HAS_TORCH:
        raise SystemExit("heads self-test requires torch (env `pyg`).")
    _self_test()
