"""Absorbed X-ray dose model.

Dose is taken from a RADDOSE-3D calculation supplied with the dataset and
converted to a per-frame quantity.  The spectrometer acquires a continuous
kinetic series at a fixed cycle time while the crystal is under continuous
illumination, so absorbed dose accumulates linearly with elapsed time and
scales with the attenuator transmission.

Two conventions matter and are easy to conflate:

``dose at frame i``
    Dose absorbed *by the end of* frame ``i``, i.e. ``(i + 1) * rate * dt``.
``dose since dark state``
    Dose accumulated since the reference frames used as the dark state.

Difference spectra are referenced to a dark state built from the first few
frames, so the second convention is the one that belongs on the x-axis of a
dose-response plot.  :func:`dose_axis` implements it explicitly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DoseModel", "dose_axis"]


class DoseModel:
    """Convert frame index and transmission to absorbed dose.

    Parameters
    ----------
    dose_per_exposure_MGy
        Absorbed dose for one RADDOSE-3D exposure at full beam.
    exposure_s
        Duration of that exposure.
    """

    def __init__(self, dose_per_exposure_MGy: float, exposure_s: float) -> None:
        if exposure_s <= 0:
            raise ValueError("exposure_s must be positive")
        self.dose_per_exposure_MGy = float(dose_per_exposure_MGy)
        self.exposure_s = float(exposure_s)

    @property
    def rate_MGy_per_s(self) -> float:
        """Dose rate at 100 % transmission."""
        return self.dose_per_exposure_MGy / self.exposure_s

    def rate_at(self, transmission_pct: float) -> float:
        """Dose rate (MGy/s) at a given nominal attenuator transmission."""
        return self.rate_MGy_per_s * (float(transmission_pct) / 100.0)

    def dose_MGy(self, time_s: np.ndarray, transmission_pct: float) -> np.ndarray:
        """Cumulative absorbed dose at each elapsed time."""
        return np.asarray(time_s, dtype=float) * self.rate_at(transmission_pct)

    @classmethod
    def from_config(cls, dose_cfg: dict) -> "DoseModel":
        return cls(
            dose_per_exposure_MGy=dose_cfg["raddose_dose_per_exposure_MGy"],
            exposure_s=dose_cfg["raddose_exposure_s"],
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"DoseModel({self.rate_MGy_per_s:.4f} MGy/s at 100% T)"


def dose_axis(
    time_s: np.ndarray,
    transmission_pct: float,
    model: DoseModel,
    dark_frames: int = 5,
) -> np.ndarray:
    """Dose accumulated since the end of the dark-state reference window.

    Parameters
    ----------
    time_s
        Elapsed time of each frame.
    transmission_pct
        Nominal attenuator transmission for this condition.
    model
        Dose model.
    dark_frames
        Number of leading frames averaged to form the dark state.

    Returns
    -------
    ndarray
        Dose in MGy, zero at the end of the dark window and negative for
        the dark frames themselves.  Keeping the dark frames on the same
        axis rather than clipping them to zero means a dose-response plot
        shows the pre-irradiation baseline honestly.
    """
    time_s = np.asarray(time_s, dtype=float)
    if dark_frames < 1:
        raise ValueError("dark_frames must be >= 1")
    t0 = time_s[min(dark_frames, time_s.size) - 1]
    return (time_s - t0) * model.rate_at(transmission_pct)
