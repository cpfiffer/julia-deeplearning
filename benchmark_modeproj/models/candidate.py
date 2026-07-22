"""
candidate.py - the CANDIDATE mode-projection delta-ML architecture (Section 3).

Two encoders, per-mode readout, decomposed heads, shared broadening.

  geometry encoder : DimeNet++ over the LF-optimised 3D geometry -> atom
                     embeddings h (N_atoms, d). Reuse Paper 1 hyperparameters.
  LF-spectrum enc  : small set-transformer over the LF stick list; each stick a
                     token (omega_k, log I_k) lifted with Fourier features on
                     omega_k -> spectrum tokens t (N_modes, d).
                     (Fallback: 1D Conv/ViT over broadened LF trace if sticks
                     are unavailable -> tag candidate arm downgraded.)
  mode-projection  : m_k = sum_i w_ik * h_i, w_ik = ||L[k,i,:]|| normalised over
     readout          atoms (the thing under test). Concatenate Fourier features
                     of omega_k and log I_k -> mode tokens m (N_modes, d).
  fusion           : cross-attention, mode tokens = queries, spectrum tokens =
                     keys/values; gated residual fusion.
  heads            : FrequencyHead / IntensityHead / ResidualHead (identity init).
  broadening       : SHARED SharedPseudoVoigt instance.

The two readout variants needed by the ablation (-modeproj global pooling) and
the input-encoding variants (-stickenc trace encoder) are selectable by flag so
the second-wave ablation reuses this same class.

Requires torch + torch_geometric (env `pyg`). Not executed in the authoring
container; the geometry-encoder call is delegated to the frozen Paper 1
DimeNet++ so parity is exact.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

    class _NNStub:  # import-safety stub; real nn used only under torch (Katana)
        Module = object
    nn = _NNStub()  # type: ignore

from .heads import FrequencyHead, IntensityHead, ResidualHead


def fourier_features(x, num_bands: int = 8, max_val: float = 4000.0):
    """Fourier positional features for a scalar (e.g. omega_k in cm^-1)."""
    x = x / max_val
    freqs = 2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype) * math.pi
    ang = x.unsqueeze(-1) * freqs  # (..., num_bands)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (..., 2*num_bands)


class StickSetTransformer(nn.Module):
    """Set-transformer over LF sticks -> spectrum tokens t (K, d)."""

    def __init__(self, d, num_bands=8, layers=2, heads=4):
        super().__init__()
        in_dim = 2 * num_bands + 1  # fourier(omega) + log I
        self.num_bands = num_bands
        self.proj = nn.Linear(in_dim, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)

    def forward(self, omega, log_i, key_padding_mask=None):
        ff = fourier_features(omega, self.num_bands)
        tok = self.proj(torch.cat([ff, log_i.unsqueeze(-1)], dim=-1))
        return self.encoder(tok, src_key_padding_mask=key_padding_mask)


class ModeProjectionReadout(nn.Module):
    """m_k = sum_i w_ik * h_i, w_ik = ||L[k,i,:]|| normalised over atoms.

    variant='modeproj' (default, under test) or 'global' (ablation -modeproj:
    replace with global pooling, same weight for every mode).
    """

    def __init__(self, d, num_bands=8, variant="modeproj"):
        super().__init__()
        assert variant in ("modeproj", "global")
        self.variant = variant
        self.num_bands = num_bands
        self.mix = nn.Linear(d + 2 * num_bands + 2 * num_bands, d)

    def forward(self, h, L_norm, omega, log_i, atom_mask=None):
        # h: (B, N, d); L_norm: (B, K, N) atom displacement amplitudes ||L[k,i]||
        if self.variant == "modeproj":
            w = L_norm
            if atom_mask is not None:
                w = w * atom_mask.unsqueeze(1)
            w = w / (w.sum(-1, keepdim=True) + 1e-8)   # normalise over atoms
            m = torch.bmm(w, h)                        # (B, K, d)
        else:  # global pooling: every mode gets the molecule-mean atom embedding
            if atom_mask is not None:
                am = atom_mask.unsqueeze(-1)
                g = (h * am).sum(1) / am.sum(1).clamp_min(1e-6)
            else:
                g = h.mean(1)
            m = g.unsqueeze(1).expand(-1, omega.shape[1], -1)
        ff_w = fourier_features(omega, self.num_bands)
        ff_i = fourier_features(log_i, self.num_bands, max_val=1.0)
        return self.mix(torch.cat([m, ff_w, ff_i], dim=-1))


class GatedCrossFusion(nn.Module):
    """Cross-attention (queries=mode tokens, keys/values=spectrum tokens) with
    gated residual fusion so the net learns per-molecule how much to trust LF."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())

    def forward(self, m, t, t_key_padding_mask=None):
        a, _ = self.attn(m, t, t, key_padding_mask=t_key_padding_mask)
        g = self.gate(torch.cat([m, a], dim=-1))
        return self.norm(m + g * a)


class ModeProjectionDeltaML(nn.Module):
    """Full CANDIDATE model. `geometry_encoder` is the frozen Paper 1 DimeNet++
    returning per-atom embeddings h (B, N, d); injected for exact parity."""

    def __init__(self, geometry_encoder, broadening, d=128,
                 readout_variant="modeproj", freq_form="mult",
                 residual_gated=True, use_stick_encoder=True):
        super().__init__()
        self.geometry_encoder = geometry_encoder
        self.broadening = broadening  # SHARED instance
        self.use_stick_encoder = use_stick_encoder
        self.freq_form = freq_form
        self.residual_gated = residual_gated

        if use_stick_encoder:
            self.spec_encoder = StickSetTransformer(d)
        else:
            # -stickenc ablation: Conv/ViT over broadened LF trace
            self.spec_encoder = TraceConvEncoder(d)

        self.readout = ModeProjectionReadout(d, variant=readout_variant)
        self.fusion = GatedCrossFusion(d)
        self.freq_head = FrequencyHead(d) if freq_form == "mult" else AdditiveFreqHead(d)
        self.int_head = IntensityHead(d)
        F = broadening.grid.numel()
        self.res_head = ResidualHead(d, F, gate_init=(-6.0 if residual_gated else 0.0))
        self._ungate_residual = not residual_gated

    def forward(self, batch):
        h = self.geometry_encoder(batch)               # (B, N, d)
        omega_lf = batch["omega_lf"]                   # (B, K)
        log_i_lf = batch["log_i_lf"]                   # (B, K)
        stick_mask = batch.get("stick_mask")           # (B, K) 1=real
        atom_mask = batch.get("atom_mask")             # (B, N)
        L_norm = batch["L_norm"]                       # (B, K, N)

        kpm = None if stick_mask is None else (stick_mask == 0)
        if self.use_stick_encoder:
            t = self.spec_encoder(omega_lf, log_i_lf, key_padding_mask=kpm)
        else:
            t = self.spec_encoder(batch["lf_trace"])   # (B, K, d) tokens over trace

        m = self.readout(h, L_norm, omega_lf, log_i_lf, atom_mask=atom_mask)
        m = self.fusion(m, t, t_key_padding_mask=kpm)

        omega_pred = self.freq_head(m, omega_lf)
        log_i_pred = self.int_head(m, log_i_lf)
        residual, gate = self.res_head(m, mask=stick_mask)
        if self._ungate_residual:
            residual = residual / gate.clamp_min(1e-6)  # remove gating (ablation)

        spec = self.broadening(omega_pred, log_i_pred.exp(), mask=stick_mask)
        spec = (spec + residual).clamp_min(0.0)
        return {"spectrum": spec, "omega_pred": omega_pred,
                "log_i_pred": log_i_pred, "residual_gate": gate}


class AdditiveFreqHead(nn.Module):
    """-multwarp ablation: additive frequency shift instead of bounded mult."""

    def __init__(self, d, hidden=None, max_shift_cm: float = 100.0):
        super().__init__()
        from .heads import _mlp
        self.max_shift = max_shift_cm
        self.mlp = _mlp(d, hidden or d, 1, zero_init_last=True)

    def forward(self, m, omega_lf):
        return omega_lf + self.max_shift * torch.tanh(self.mlp(m).squeeze(-1))


class TraceConvEncoder(nn.Module):
    """Fallback / -stickenc: 1D Conv over the broadened LF trace -> tokens."""

    def __init__(self, d, n_tokens=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, d, 15, stride=2, padding=7), nn.SiLU(),
            nn.Conv1d(d, d, 15, stride=2, padding=7), nn.SiLU(),
            nn.AdaptiveAvgPool1d(n_tokens),
        )

    def forward(self, trace):
        z = self.net(trace.unsqueeze(1))  # (B, d, n_tokens)
        return z.transpose(1, 2)          # (B, n_tokens, d)
