"""End-to-end processing of one condition, from raw files to difference spectra.

The processing order matters and is fixed here so that every figure and
table in the analysis derives from an identical treatment:

1. **Read** each film position's kinetic series.
2. **Detect resets**, stitch isolated ones, truncate at the first dense
   burst (:mod:`crystalspec.resets`).
3. **Locate the shutter** (:mod:`crystalspec.onset`).  Acquisition began
   by hand seconds before irradiation, by a delay that differs between
   traces and was not recorded.
4. **Reference to the dark state**: subtract the mean of the frames
   immediately preceding the shutter, giving :math:`\\Delta A`.
5. **Check optical validity** (:mod:`crystalspec.optical`) and cut where
   the sample stops being a valid absorbance sample.  This precedes
   scatter removal because the power-law fit would absorb the broadband
   offset that signals the damage.
6. **Remove scattering** by fitting a power law to the non-absorbing red
   region (:mod:`crystalspec.scatter`).

Dose accumulates from the shutter opening.  Zeroing it at frame 0 instead
attributes dose that was never delivered, and because the error is a fixed
*time* offset it scales with dose rate: a few tenths of a MGy at 5 %
transmission but several MGy at 100 %.

Steps 3 and 4 commute to within numerical precision -- the scatter model is
linear and fitted per spectrum -- but referencing first means the scatter
fit acts on a quantity already centred near zero, so the fitted offset
absorbs only the *change* in scattering rather than its absolute level.

Crystals within a condition are combined only at the last moment, and
always as a mean plus the spread across crystals, never as a mean of raw
absorbances: the raw level is dominated by crystal size (see
:mod:`crystalspec.scatter`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dose import DoseModel
from .onset import OnsetReport, detect_onset
from .optical import OpticalReport, check_optical_validity
from .registry import Condition, Registry
from .resets import ResetReport, correct_resets
from .scatter import DEFAULT_EXPONENT, DEFAULT_FIT_WINDOW, remove_scatter

__all__ = ["CrystalTrace", "ConditionResult", "process_condition"]

DEFAULT_DARK_FRAMES = 5


@dataclass
class CrystalTrace:
    """Processed data for one crystal position."""

    filename: str
    wavelength_nm: np.ndarray
    delta_a: np.ndarray            # (n_wavelength, n_frames) scatter-corrected
    time_s: np.ndarray             # (n_frames,)
    dose_MGy: np.ndarray           # (n_frames,) since shutter opening
    raw_dark: np.ndarray           # (n_wavelength,) mean raw dark spectrum
    reset_report: ResetReport
    onset_report: OnsetReport
    optical_report: OpticalReport

    @property
    def n_frames(self) -> int:
        return int(self.delta_a.shape[1])


@dataclass
class ConditionResult:
    """All crystals of one condition on a common frame grid.

    Attributes
    ----------
    condition
        The condition processed.
    wavelength_nm
        Wavelength axis, shape ``(n_wavelength,)``.
    delta_a
        Stack of per-crystal difference spectra, shape
        ``(n_crystals, n_wavelength, n_frames)``, padded with NaN where a
        crystal's trace ended earlier than the longest in the condition.
    time_s, dose_MGy
        Common axes, shape ``(n_frames,)``.
    n_contributing
        Number of crystals with data at each frame, shape ``(n_frames,)``.
    min_crystals
        Frames with fewer contributing crystals than this are not reported
        by :meth:`valid`; averaging over a shrinking, non-random subset of
        crystals would otherwise put a step in the mean at each dropout.
    traces
        The individual :class:`CrystalTrace` objects, at full length.
    """

    condition: Condition
    wavelength_nm: np.ndarray
    delta_a: np.ndarray
    time_s: np.ndarray
    dose_MGy: np.ndarray
    n_contributing: np.ndarray
    traces: list[CrystalTrace]
    min_crystals: int = 3

    @property
    def n_crystals(self) -> int:
        return int(self.delta_a.shape[0])

    def valid(self) -> np.ndarray:
        """Boolean mask over frames with enough contributing crystals."""
        return self.n_contributing >= self.min_crystals

    def mean(self) -> np.ndarray:
        """Mean difference spectrum across crystals, ``(n_wavelength, n_frames)``.

        Frames failing :meth:`valid` are returned as NaN so they cannot
        silently enter a fit or a plot.
        """
        with np.errstate(invalid="ignore"):
            m = np.nanmean(self.delta_a, axis=0)
        m[:, ~self.valid()] = np.nan
        return m

    def sem(self) -> np.ndarray:
        """Standard error across crystals, ``(n_wavelength, n_frames)``.

        With at most five crystals this is a coarse estimate, and it
        describes the spread of an ensemble of independent crystal
        positions rather than a measurement precision.
        """
        if self.n_crystals < 2:
            return np.zeros_like(self.mean())
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(self.delta_a, axis=0, ddof=1)
        out = sd / np.sqrt(np.maximum(self.n_contributing, 1))[None, :]
        out[:, ~self.valid()] = np.nan
        return out

    def at_dose(self, dose_MGy: float) -> np.ndarray:
        """Mean difference spectrum at the valid frame nearest a given dose."""
        valid = self.valid()
        if not valid.any():
            raise ValueError(f"{self.condition.key}: no frames with enough crystals")
        candidates = np.flatnonzero(valid)
        idx = candidates[np.argmin(np.abs(self.dose_MGy[candidates] - dose_MGy))]
        return self.mean()[:, idx]

    def max_valid_dose_MGy(self) -> float:
        """Highest dose at which the condition still has enough crystals."""
        return float(self.dose_MGy[self.valid()].max())


def process_condition(
    registry: Registry,
    key: str,
    dose_model: DoseModel,
    dark_frames: int = DEFAULT_DARK_FRAMES,
    scatter_exponent: float = DEFAULT_EXPONENT,
    scatter_window: tuple[float, float] = DEFAULT_FIT_WINDOW,
    truncate: bool = True,
    min_crystals: int = 3,
) -> ConditionResult:
    """Process every crystal of one condition.

    Parameters
    ----------
    registry
        Experiment registry.
    key
        Condition key, e.g. ``"DTNB_5"``.
    dose_model
        Converts elapsed time to absorbed dose.
    dark_frames
        Leading frames averaged to form each crystal's dark state.
    scatter_exponent, scatter_window
        Passed to :func:`crystalspec.scatter.remove_scatter`.
    truncate
        Cut each trace at the first dense reset burst.
    min_crystals
        Minimum number of crystals contributing to a frame for it to be
        reported in the condition mean.

    Returns
    -------
    ConditionResult

    Raises
    ------
    ValueError
        If a crystal has fewer usable frames than ``dark_frames``.
    """
    condition = registry.conditions[key]
    traces: list[CrystalTrace] = []

    for filename in condition.files:
        acq = registry.load(filename)
        cycle = acq.cycle_time_s or 0.1

        stitched, report = correct_resets(
            acq.wavelength_nm, acq.absorbance, cycle_time_s=cycle, truncate=truncate
        )
        if stitched.shape[1] <= dark_frames:
            raise ValueError(
                f"{filename}: only {stitched.shape[1]} usable frames, "
                f"need more than dark_frames={dark_frames}"
            )

        # Locate the shutter.  Acquisition was started by hand seconds
        # before irradiation began, and the delay differs between traces.
        onset = detect_onset(acq.wavelength_nm, stitched, cycle_time_s=cycle)
        start = onset.frame
        if start >= dark_frames:
            dark = stitched[:, start - dark_frames : start].mean(axis=1)
        else:
            # No pre-shutter window resolved: fall back to the leading frames
            # and accept that the dark state may already be irradiated.
            dark = stitched[:, :dark_frames].mean(axis=1)
            start = 0

        delta_raw = stitched[:, start:] - dark[:, None]

        # Optical validity is judged BEFORE scatter removal: the power-law
        # fit would absorb the very broadband offset that signals damage.
        optical = check_optical_validity(acq.wavelength_nm, delta_raw)
        if truncate and optical.compromised:
            delta_raw = delta_raw[:, : optical.n_valid]
        if delta_raw.shape[1] == 0:
            raise ValueError(f"{filename}: no optically valid frames after onset")

        delta = remove_scatter(
            acq.wavelength_nm, delta_raw, exponent=scatter_exponent,
            window=scatter_window,
        )

        # Dose accumulates from the shutter opening, not from frame zero.
        time_s = np.arange(delta.shape[1]) * cycle
        rate = dose_model.rate_at(condition.transmission_pct)
        traces.append(
            CrystalTrace(
                filename=filename,
                wavelength_nm=acq.wavelength_nm,
                delta_a=delta,
                time_s=time_s + start * cycle,
                dose_MGy=time_s * rate,
                raw_dark=dark,
                reset_report=report,
                onset_report=onset,
                optical_report=optical,
            )
        )

    wavelength = traces[0].wavelength_nm
    for t in traces[1:]:
        if not np.allclose(t.wavelength_nm, wavelength):
            raise ValueError(f"{t.filename}: wavelength axis differs within condition")

    # Pad to the longest trace rather than cropping to the shortest: one
    # crystal destabilising early should not discard the others' data.
    n = max(t.n_frames for t in traces)
    longest = max(traces, key=lambda t: t.n_frames)
    stack = np.full((len(traces), wavelength.size, n), np.nan)
    for i, t in enumerate(traces):
        stack[i, :, : t.n_frames] = t.delta_a
    n_contributing = np.isfinite(stack[:, 0, :]).sum(axis=0)

    return ConditionResult(
        condition=condition,
        wavelength_nm=wavelength,
        delta_a=stack,
        time_s=longest.time_s,
        dose_MGy=longest.dose_MGy,
        n_contributing=n_contributing,
        traces=traces,
        min_crystals=min_crystals,
    )
