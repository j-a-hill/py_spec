"""Quantitation of the difference signal.

The TNB band is a shoulder in crystallo rather than a resolved peak, so it
cannot be quantified by fitting a peak to it.  This module provides the
model-free route -- integrate the difference signal over a fixed window --
which makes no assumption about band shape and is the primary quantity in
the figures.  Two further routes (singular value decomposition and
reference-spectrum unmixing) are cross-checks reported alongside it.

The internal-zero window is the control: after scatter removal the
difference signal there should be zero at every dose, so whatever it does
instead bounds the precision of every other band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BandSeries", "integrate_band", "band_series", "bootstrap_ci"]


@dataclass
class BandSeries:
    """Integrated difference signal in one window, versus dose.

    Attributes
    ----------
    dose_MGy
        Dose axis, shape ``(n_frames,)``.
    per_crystal
        Integrated signal for each crystal, shape ``(n_crystals, n_frames)``,
        NaN where that crystal had no data.
    mean, lo, hi
        Across-crystal mean and confidence bounds, shape ``(n_frames,)``.
    window
        The wavelength window integrated, nm.
    name
        Window name.
    """

    dose_MGy: np.ndarray
    per_crystal: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    window: tuple[float, float]
    name: str

    @property
    def n_crystals(self) -> int:
        return int(self.per_crystal.shape[0])


def integrate_band(
    wavelength_nm: np.ndarray,
    spectra: np.ndarray,
    window: tuple[float, float],
    normalise: bool = True,
) -> np.ndarray:
    """Integrate spectra over a wavelength window.

    Parameters
    ----------
    wavelength_nm
        Wavelength axis, shape ``(n_wavelength,)``.
    spectra
        Shape ``(..., n_wavelength, n_frames)`` or ``(n_wavelength, ...)``;
        integration is over the axis matching ``wavelength_nm``.
    window
        ``(low, high)`` in nm.
    normalise
        Divide by the window width, giving a mean absorbance change rather
        than an area.  This keeps values comparable between windows of
        different width and in the same units as the spectra themselves.

    Returns
    -------
    ndarray
        Integrated signal with the wavelength axis removed.

    Raises
    ------
    ValueError
        If fewer than two points fall inside the window.
    """
    w = np.asarray(wavelength_nm, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    axis = next(
        (i for i, n in enumerate(spectra.shape) if n == w.size),
        None,
    )
    if axis is None:
        raise ValueError(
            f"no axis of {spectra.shape} matches wavelength length {w.size}"
        )

    mask = (w >= window[0]) & (w <= window[1])
    if mask.sum() < 2:
        raise ValueError(f"window {window} contains {mask.sum()} points; need >= 2")

    sub = np.take(spectra, np.flatnonzero(mask), axis=axis)
    area = np.trapezoid(sub, w[mask], axis=axis)
    return area / (w[mask][-1] - w[mask][0]) if normalise else area


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap confidence bounds over the first axis.

    Resampling crystals -- the unit of replication -- rather than assuming
    normality, because five values is far too few for a t-interval to be
    trustworthy.  The interval is still wide and should be read as such.

    Parameters
    ----------
    values
        Shape ``(n_crystals, n_frames)``; NaN entries are ignored.
    n_boot
        Bootstrap replicates.
    alpha
        Two-sided significance level.
    seed
        Random seed, so figures are reproducible.

    Returns
    -------
    lo, hi : ndarray
        Confidence bounds, shape ``(n_frames,)``.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    with np.errstate(invalid="ignore"):
        boots = np.nanmean(values[idx], axis=1)
        lo = np.nanpercentile(boots, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def band_series(
    result,
    window: tuple[float, float],
    name: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> BandSeries:
    """Integrated difference signal versus dose for one condition.

    Parameters
    ----------
    result
        A :class:`~crystalspec.pipeline.ConditionResult`.
    window
        Wavelength window, nm.
    name
        Label for the window.
    n_boot, seed
        Passed to :func:`bootstrap_ci`.

    Returns
    -------
    BandSeries
        Restricted to frames with enough contributing crystals.
    """
    valid = result.valid()
    per_crystal = integrate_band(
        result.wavelength_nm, result.delta_a[:, :, valid], window
    )
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(per_crystal, axis=0)
    lo, hi = bootstrap_ci(per_crystal, n_boot=n_boot, seed=seed)
    return BandSeries(
        dose_MGy=result.dose_MGy[valid],
        per_crystal=per_crystal,
        mean=mean,
        lo=lo,
        hi=hi,
        window=window,
        name=name,
    )
