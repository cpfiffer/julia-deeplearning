"""
broadening.py - the SHARED learnable pseudo-Voigt broadening layer.

Matched-comparison control #3: BOTH arms broaden their corrected sticks through
the SAME pseudo-Voigt layer so broadening is not a confound. The comparison
isolates the encoder + readout + correction heads only.

The intended usage is a SINGLE instance passed to both arms' models (or, in the
independent-training setting, an identically-initialised-and-configured instance
with the same learnable-parameter semantics). The layer maps a variable-length
stick list (frequencies omega_k in cm^-1, intensities I_k) onto the fixed
evaluation grid.

Requires torch. This file is part of the Katana-side package (env `pyg`) and is
not executed in the authoring container.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch absent in authoring container
    _HAS_TORCH = False

    class _NNStub:  # import-safety stub; real nn used only under torch (Katana)
        Module = object
    nn = _NNStub()  # type: ignore


class SharedPseudoVoigt(nn.Module):
    """Learnable pseudo-Voigt broadening onto a fixed grid.

    A pseudo-Voigt line is eta * Lorentzian + (1 - eta) * Gaussian, with a
    shared learnable width (sigma/gamma) and mixing eta. Widths are kept
    positive via softplus; eta via sigmoid. One instance is shared by both arms.

    Parameters
    ----------
    grid : 1-D tensor of cm^-1 sample points (registered as a buffer, not
           learned) so both arms broaden onto byte-identical support.
    """

    def __init__(self, grid, init_width_cm: float = 8.0, init_eta: float = 0.5):
        if not _HAS_TORCH:
            raise RuntimeError("SharedPseudoVoigt requires torch (env `pyg`).")
        super().__init__()
        grid = torch.as_tensor(grid, dtype=torch.float32)
        self.register_buffer("grid", grid)  # (F,)
        # softplus^{-1}(init_width) so the initial effective width is init_width
        inv_sp = math.log(math.expm1(max(init_width_cm, 1e-3)))
        self._raw_width = nn.Parameter(torch.tensor(float(inv_sp)))
        # logit(init_eta)
        self._raw_eta = nn.Parameter(torch.tensor(math.log(init_eta / (1 - init_eta))))

    @property
    def width(self):
        return torch.nn.functional.softplus(self._raw_width) + 1e-3

    @property
    def eta(self):
        return torch.sigmoid(self._raw_eta)

    def forward(self, omega, intensity, mask=None):
        """Broaden a batch of stick lists onto the grid.

        omega, intensity : (B, K) padded stick frequencies (cm^-1) and
                           non-negative intensities.
        mask             : (B, K) bool/float; 1 for real sticks, 0 for padding.
        returns          : (B, F) non-negative broadened spectra on self.grid.
        """
        w = self.width
        eta = self.eta
        # (B, K, 1) - (F,) -> (B, K, F)
        d = omega.unsqueeze(-1) - self.grid.view(1, 1, -1)
        gauss = torch.exp(-0.5 * (d / w) ** 2)
        lorentz = 1.0 / (1.0 + (d / w) ** 2)
        line = eta * lorentz + (1.0 - eta) * gauss  # (B, K, F)
        amp = intensity.clamp_min(0.0).unsqueeze(-1)  # (B, K, 1)
        if mask is not None:
            amp = amp * mask.to(amp.dtype).unsqueeze(-1)
        spec = (amp * line).sum(dim=1)  # (B, F)
        return spec.clamp_min(0.0)

    def extra_repr(self):
        return f"F={self.grid.numel()}, shared=True"
