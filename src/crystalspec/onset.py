"""Locate the X-ray shutter opening within each kinetic series.

Acquisition was started by hand a few seconds before the shutter was
opened, and the delay was not recorded.  Every trace therefore begins with
a variable run of genuinely dark frames.  Treating frame 0 as the start of
irradiation attributes dose that was never delivered, and because the
error is a fixed *time* offset it converts into a dose error proportional
to the dose rate: negligible at 5 % transmission, several MGy at 100 %.

The shutter is located from the data, as the earliest persistent departure
in *any* spectral region.  An earlier version triggered on the
300--325 nm radiation-damage band alone; that band responds slowly at low
dose rate, so at 5 % transmission it cleared threshold about 10 s after
irradiation had demonstrably begun -- by which point the TNB window had
already fallen by 0.02 absorbance units.  Referencing to that late window
subtracted part of the signal and inverted the sign of the result.  A
detector that must not miss the true start therefore cannot rely on any
single band: whichever region responds first defines the onset.

Because the dark reference is taken immediately before the detected
onset, a *late* onset is far more damaging than an early one -- it places
the reference on already-irradiated frames.  The threshold is
correspondingly loose, and the reported onset should be read as an upper
bound on the true shutter time.

Detection is deliberately conservative in two ways.  A departure must
persist for ``run`` consecutive frames to count, so a single noisy frame
or an isolated detector reset cannot trigger it.  The threshold is set
from the noise of the leading segment itself rather than from a global
constant, so a trace with a poor signal-to-noise ratio simply yields a
later onset rather than a spurious early one.

**On circularity.**  The onset is fitted per trace from the same signal
whose dose dependence is later analysed, so recovering a plausible onset
is not by itself evidence that the dose axis is right.  The independent
check is that onsets fitted separately for traces at different
transmissions bring their dose-response curves into agreement: the
detector operates on each trace in isolation and has no mechanism for
arranging a collapse across conditions it never sees together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OnsetReport", "detect_onset", "DEFAULT_BANDS", "DEFAULT_SIGMA",
           "DEFAULT_RUN", "DEFAULT_BASELINE_FRAMES"]

DEFAULT_BANDS = (
    (285.0, 300.0),
    (300.0, 325.0),
    (325.0, 360.0),
    (360.0, 395.0),
    (395.0, 430.0),
    (430.0, 500.0),
    (600.0, 700.0),
)
DEFAULT_SIGMA = 6.0
DEFAULT_RUN = 3
DEFAULT_BASELINE_FRAMES = 5
_NOISE_FLOOR = 3e-4


@dataclass(frozen=True)
class OnsetReport:
    """Where irradiation began in one kinetic series.

    Attributes
    ----------
    frame
        Index of the first irradiated frame.  Zero means no pre-shutter
        window was detected and the trace is treated as irradiated
        throughout.
    time_s
        ``frame`` expressed in seconds.
    baseline
        Mean band absorbance over the frames preceding ``frame``.
    noise
        Noise estimate used to set the detection threshold.
    detected
        False when the band never departs persistently from baseline, in
        which case ``frame`` falls back to 0 and no dose is discarded.
    n_dark_frames
        Number of frames available as a dark reference.
    """

    frame: int
    time_s: float
    baseline: float
    noise: float
    detected: bool
    n_dark_frames: int


def detect_onset(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    cycle_time_s: float = 0.1,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    sigma: float = DEFAULT_SIGMA,
    run: int = DEFAULT_RUN,
    baseline_frames: int = DEFAULT_BASELINE_FRAMES,
) -> OnsetReport:
    """Find the first irradiated frame of a kinetic series.

    Parameters
    ----------
    wavelength_nm
        Wavelength axis, ascending.
    absorbance
        ``(n_wavelength, n_frames)`` raw absorbance.
    cycle_time_s
        Frame period, used only to convert the index to seconds.
    bands
        Wavelength windows scanned; the earliest crossing wins.
    sigma
        Detection threshold in multiples of the leading-segment noise.
    run
        Number of consecutive frames that must exceed the threshold.
    baseline_frames
        Frames used for the initial baseline estimate.

    Returns
    -------
    OnsetReport
    """
    if absorbance.shape[1] <= baseline_frames + run:
        return OnsetReport(0, 0.0, float("nan"), float("nan"), False,
                           int(absorbance.shape[1]))

    best: tuple[int, float, float] | None = None
    for lo, hi in bands:
        mask = (wavelength_nm >= lo) & (wavelength_nm < hi)
        if not mask.any():
            continue
        series = absorbance[mask].mean(axis=0)
        baseline = float(series[:baseline_frames].mean())

        lead = series[: min(40, series.size)]
        noise = max(
            float(series[:baseline_frames].std()),
            float(np.diff(lead).std() / np.sqrt(2)) if lead.size > 1 else 0.0,
            _NOISE_FLOOR,
        )

        exceeds = np.abs(series - baseline) > sigma * noise
        streak = 0
        for i, flag in enumerate(exceeds):
            streak = streak + 1 if flag else 0
            if streak >= run:
                frame = i - run + 1
                if best is None or frame < best[0]:
                    best = (int(frame), baseline, noise)
                break

    if best is None:
        return OnsetReport(0, 0.0, float("nan"), float("nan"), False,
                           int(absorbance.shape[1]))

    frame, baseline, noise = best
    if frame < baseline_frames:
        # Signal already moving in the first frames: no clean dark window
        # exists, so keep frame 0 and accept the leading frames as-is.
        return OnsetReport(0, 0.0, baseline, noise, False, int(absorbance.shape[1]))

    return OnsetReport(
        frame=frame,
        time_s=float(frame * cycle_time_s),
        baseline=baseline,
        noise=noise,
        detected=True,
        n_dark_frames=frame,
    )
