"""Detection and repair of spectrometer baseline resets.

During acquisition the detector baseline drifts upwards, and the operator
periodically resets it.  A reset appears in the data as a step: the whole
spectrum shifts between two consecutive frames, with no intervening
transition.  Left in place, these steps corrupt any fit through them --
they were the origin of the non-physical "rebound" in earlier versions of
this analysis, where a decay curve was fitted straight across a reset.

Detection
---------
A reset displaces the *entire* spectrum, so it is detected on a broadband
average (500-650 nm by default) where no chromophore absorbs and the
photon flux is high.  In conditions with no resets at all (5 % and 25 %
transmission) the frame-to-frame change of that average has a robust
standard deviation of 3.2e-4 absorbance units, so the default threshold of
0.01 sits roughly 30 sigma above the noise and does not fire on any of the
20 crystals in those conditions.

Repair: stitch or truncate
--------------------------
Across a reset the offset is smooth and almost wavelength-independent above
~400 nm, so an isolated reset can be *stitched*: the wavelength-resolved
difference between the frames bracketing the step is subtracted from
everything after it, restoring a continuous trace.

Resets do not, however, arrive uniformly.  They are isolated early in an
acquisition and then arrive in dense bursts once the baseline becomes
unstable -- in the 100 % transmission apo series, for instance, single
steps around 17-20 s are followed by dozens within a few seconds.  Once
that happens the trace is no longer a continuous measurement of one sample
state and stitching it would manufacture a smooth curve from unrelated
segments.  A burst therefore *ends* the usable window instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ResetReport", "detect_resets", "stitch_resets", "correct_resets"]

DEFAULT_THRESHOLD = 0.01
DEFAULT_BAND = (500.0, 650.0)
#: A burst is this many resets within ``burst_window_s``.
DEFAULT_BURST_COUNT = 3
DEFAULT_BURST_WINDOW_S = 2.0
#: Frames averaged either side of a step to estimate its offset.
DEFAULT_BRACKET = 5


@dataclass
class ResetReport:
    """Where the resets are and how much of the trace survives them.

    Attributes
    ----------
    step_frames
        Frame indices at which a reset was detected; frame ``k`` is the
        first frame *after* the step.
    step_times_s
        The same, in seconds.
    burst_start_frame
        First frame of the first dense burst, or ``None`` if there is none.
    usable_frames
        Number of leading frames that may be analysed.  Equal to the burst
        start when a burst exists, otherwise the full trace.
    stitched_frames
        Steps that were repaired by stitching (those inside the usable
        window).
    threshold
        Detection threshold used.
    """

    step_frames: np.ndarray
    step_times_s: np.ndarray
    burst_start_frame: int | None
    usable_frames: int
    stitched_frames: np.ndarray = field(default_factory=lambda: np.array([], int))
    threshold: float = DEFAULT_THRESHOLD

    @property
    def n_steps(self) -> int:
        return int(self.step_frames.size)

    @property
    def truncated(self) -> bool:
        return self.burst_start_frame is not None

    def usable_time_s(self, cycle_time_s: float) -> float:
        return self.usable_frames * cycle_time_s

    def __repr__(self) -> str:  # pragma: no cover - display only
        state = (
            f"truncated at frame {self.burst_start_frame}"
            if self.truncated
            else "no truncation"
        )
        return (
            f"ResetReport({self.n_steps} steps, {self.stitched_frames.size} stitched, "
            f"{state}, {self.usable_frames} usable frames)"
        )


def _broadband(
    absorbance: np.ndarray, wavelength_nm: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    mask = (wavelength_nm >= band[0]) & (wavelength_nm <= band[1])
    if not mask.any():
        raise ValueError(f"no wavelengths inside detection band {band}")
    return absorbance[mask, :].mean(axis=0)


def detect_resets(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    cycle_time_s: float = 0.1,
    threshold: float = DEFAULT_THRESHOLD,
    band: tuple[float, float] = DEFAULT_BAND,
    burst_count: int = DEFAULT_BURST_COUNT,
    burst_window_s: float = DEFAULT_BURST_WINDOW_S,
) -> ResetReport:
    """Locate baseline resets and the point where the trace becomes unusable.

    Parameters
    ----------
    wavelength_nm
        Wavelength axis.
    absorbance
        Spectra, shape ``(n_wavelength, n_frames)``.
    cycle_time_s
        Seconds per frame.
    threshold
        Frame-to-frame change in the broadband average that counts as a
        reset, in absorbance units.
    band
        Wavelength range used for the broadband average.
    burst_count, burst_window_s
        ``burst_count`` resets falling within ``burst_window_s`` mark the
        onset of baseline instability and end the usable window.

    Returns
    -------
    ResetReport
    """
    absorbance = np.asarray(absorbance, dtype=float)
    n_frames = absorbance.shape[1]
    broad = _broadband(absorbance, np.asarray(wavelength_nm, float), band)
    jumps = np.abs(np.diff(broad))
    steps = np.flatnonzero(jumps > threshold) + 1  # first frame after the step

    burst_start: int | None = None
    if steps.size >= burst_count:
        span = int(round(burst_window_s / cycle_time_s))
        # The first index where `burst_count` consecutive detections fall
        # inside one window marks the onset of instability.
        for i in range(steps.size - burst_count + 1):
            if steps[i + burst_count - 1] - steps[i] <= span:
                burst_start = int(steps[i])
                break

    usable = n_frames if burst_start is None else burst_start
    return ResetReport(
        step_frames=steps,
        step_times_s=steps * cycle_time_s,
        burst_start_frame=burst_start,
        usable_frames=int(usable),
        stitched_frames=steps[steps < usable],
        threshold=float(threshold),
    )


def stitch_resets(
    absorbance: np.ndarray,
    report: ResetReport,
    bracket: int = DEFAULT_BRACKET,
) -> np.ndarray:
    """Remove reset offsets by subtracting the step at each reset.

    For each reset the wavelength-resolved offset is estimated as the
    difference between the mean of ``bracket`` frames after the step and
    ``bracket`` frames before it, excluding the frames immediately adjacent
    to the step, which are often partially affected.  That offset is then
    subtracted from every subsequent frame.

    Parameters
    ----------
    absorbance
        Spectra, shape ``(n_wavelength, n_frames)``.
    report
        Output of :func:`detect_resets`; only ``stitched_frames`` is used.
    bracket
        Frames averaged on each side of a step.

    Returns
    -------
    ndarray
        Stitched spectra; the input is not modified.
    """
    out = np.array(absorbance, dtype=float, copy=True)
    n_frames = out.shape[1]

    for k in report.stitched_frames:
        k = int(k)
        pre_lo, pre_hi = max(0, k - bracket - 1), max(1, k - 1)
        post_lo, post_hi = min(k + 1, n_frames - 1), min(k + 1 + bracket, n_frames)
        if pre_hi <= pre_lo or post_hi <= post_lo:
            continue
        offset = out[:, post_lo:post_hi].mean(axis=1) - out[:, pre_lo:pre_hi].mean(axis=1)
        out[:, k:] -= offset[:, None]
    return out


def correct_resets(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    cycle_time_s: float = 0.1,
    truncate: bool = True,
    **kwargs,
) -> tuple[np.ndarray, ResetReport]:
    """Detect resets, stitch the isolated ones, and truncate at a burst.

    Parameters
    ----------
    wavelength_nm, absorbance, cycle_time_s
        As for :func:`detect_resets`.
    truncate
        If true (default) the returned array is cut at the start of the
        first dense burst.  If false the full trace is returned, still
        stitched up to the burst; the report records where the cut would
        have been.
    **kwargs
        Passed to :func:`detect_resets`.

    Returns
    -------
    corrected : ndarray
    report : ResetReport
    """
    report = detect_resets(
        wavelength_nm, absorbance, cycle_time_s=cycle_time_s, **kwargs
    )
    stitched = stitch_resets(absorbance, report)
    if truncate:
        stitched = stitched[:, : report.usable_frames]
    return stitched, report
