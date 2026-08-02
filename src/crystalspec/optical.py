"""Flag frames where the sample has stopped being a valid absorbance sample.

The room-temperature exposures were deliberately pushed until the EVAL
film burned or crystals fractured.  Beyond that point the trace is still
recorded, but it no longer measures absorbance: the light path has changed
physically.

The 600--700 nm window supplies the diagnostic.  No component of this
system has a band there, and the change observed at high dose rate is flat
to within 0.001 absorbance units across 575--700 nm -- a featureless
offset, with none of the curvature a real chromophore would show.  It
therefore reports bulk opacity: bubbles, cracks and delamination.

Two properties make it a usable criterion rather than a arbitrary cut.
The magnitude separates by *dose rate* rather than by dose: referenced to
the shutter opening and compared at matched dose, the apo control at
100 % transmission reaches 0.10 absorbance units by 5 MGy while the 25 %
control is at 0.0004 at the same dose.  A trace also never recovers, so
the first sustained crossing marks the end of the usable window in the
same way a reset burst does.

**What this criterion cannot establish.**  A flat feature rules out a
*band*, not every conceivable absorber -- a genuinely structureless
absorber would be indistinguishable from opacity by this test.  Solvated
electrons absorb broadly near 700 nm but survive nanoseconds at room
temperature and cannot sustain a steady offset of this size.  The
attribution to physical damage is strongly supported, not proven, and it
is corroborated by the operator's own account of burning the film.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OpticalReport", "check_optical_validity", "DEFAULT_BAND",
           "DEFAULT_THRESHOLD", "DEFAULT_RUN"]

DEFAULT_BAND = (600.0, 700.0)
DEFAULT_THRESHOLD = 0.03
DEFAULT_RUN = 5


@dataclass(frozen=True)
class OpticalReport:
    """Optical integrity of one kinetic series.

    Attributes
    ----------
    first_bad_frame
        Index of the first frame of the sustained crossing, or ``None``
        if the trace never crosses.
    n_valid
        Number of frames before that point.
    compromised
        Whether a crossing was found.
    max_excursion
        Largest absolute red-band deviation reached.
    band, threshold
        Settings used, recorded so a report is self-describing.
    """

    first_bad_frame: int | None
    n_valid: int
    compromised: bool
    max_excursion: float
    band: tuple[float, float]
    threshold: float


def check_optical_validity(
    wavelength_nm: np.ndarray,
    delta_a: np.ndarray,
    band: tuple[float, float] = DEFAULT_BAND,
    threshold: float = DEFAULT_THRESHOLD,
    run: int = DEFAULT_RUN,
) -> OpticalReport:
    """Find where a difference trace stops being optically valid.

    Parameters
    ----------
    wavelength_nm
        Wavelength axis, ascending.
    delta_a
        ``(n_wavelength, n_frames)`` difference spectra, already
        referenced to the pre-shutter dark state and *not* scatter
        corrected -- the scatter fit would remove the very offset this
        function looks for.
    band
        Non-absorbing window used as the diagnostic.
    threshold
        Absolute deviation marking loss of optical integrity.
    run
        Consecutive frames required, so a single reset cannot trigger it.

    Returns
    -------
    OpticalReport
    """
    mask = (wavelength_nm >= band[0]) & (wavelength_nm <= band[1])
    if not mask.any() or delta_a.shape[1] == 0:
        return OpticalReport(None, int(delta_a.shape[1]), False, float("nan"),
                             band, threshold)

    level = delta_a[mask].mean(axis=0)
    exceeds = np.abs(level) > threshold

    streak = 0
    for i, flag in enumerate(exceeds):
        streak = streak + 1 if flag else 0
        if streak >= run:
            first = i - run + 1
            return OpticalReport(int(first), int(first), True,
                                 float(np.nanmax(np.abs(level))), band, threshold)

    return OpticalReport(None, int(level.size), False,
                         float(np.nanmax(np.abs(level))), band, threshold)
