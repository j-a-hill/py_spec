"""Removal of the crystal scattering background.

A microcrystal in a thin film scatters far more light than it absorbs, and
the scattered fraction rises steeply towards the blue.  Between crystals the
scattering magnitude varies by a factor of two to four even within one
experimental condition, which is why averaging raw spectra across crystal
positions is dominated by crystal size and shape rather than by chemistry.

Above about 470 nm nothing in this system absorbs -- neither TNB (412 nm
maximum), nor DTNB, nor the protein -- so the measured signal there is pure
scattering plus a wavelength-independent offset.  Fitting

.. math:: A_\\mathrm{scatter}(\\lambda) = c\\,(\\lambda / 500)^{-n} + b

on that region and subtracting it across the full range removes the
background without reference to any absorbing species.

Choice of exponent
------------------
Held-out cross-validation (fit on 500-725 nm, predict 470-500 nm) over all
40 room-temperature crystals at four timepoints gives median RMSE:

===============  ==========
model            RMSE
===============  ==========
free ``n``       0.00546
``n = 3``        0.00560
``n = 2``        0.00571
``n = 1``        0.00585
``n = 4``        0.00586
linear           0.00600
constant         0.01215
===============  ==========

Any power law beats a linear or constant background, but the exponent is
only weakly determined, and fixing ``n = 3`` costs almost nothing relative
to fitting it.  A free fit is also unstable: over the same 160 spectra,
with ``n`` bounded to ``[0, 8]``, the recovered exponent spans the whole
range (DTNB median 1.98, apo median 0.60) and lands *on* a bound for 20 %
of DTNB and 36 % of apo spectra -- the weakly scattering apo crystals
collapsing to ``n = 0`` in a third of cases, which is no scattering model
at all.

The exponent is therefore *fixed* by default, and because integrated band
values shift by up to ~23 % (TNB window) across ``n = 1..4``, that spread is
carried explicitly as a systematic uncertainty by
:func:`exponent_sensitivity` rather than being hidden in a point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ScatterModel",
    "fit_scatter",
    "remove_scatter",
    "exponent_sensitivity",
]

DEFAULT_EXPONENT = 3.0
DEFAULT_FIT_WINDOW = (470.0, 725.0)
#: Exponents spanned when reporting systematic uncertainty.
SENSITIVITY_EXPONENTS = (1.0, 2.0, 3.0, 4.0)


@dataclass
class ScatterModel:
    """Fitted scattering background for one or more spectra.

    Attributes
    ----------
    exponent
        Fixed power-law exponent ``n``.
    amplitude
        Coefficient ``c``, one per spectrum.
    offset
        Coefficient ``b``, one per spectrum.
    window
        Wavelength range used for the fit.
    """

    exponent: float
    amplitude: np.ndarray
    offset: np.ndarray
    window: tuple[float, float]

    def evaluate(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Background at each wavelength, shape ``(n_wavelength, n_spectra)``."""
        basis = _basis(wavelength_nm, self.exponent)
        coef = np.vstack([self.amplitude, self.offset])
        return basis @ coef


def _basis(wavelength_nm: np.ndarray, exponent: float) -> np.ndarray:
    """Design matrix ``[(lambda/500)^-n, 1]``."""
    w = np.asarray(wavelength_nm, dtype=float)
    return np.column_stack([(w / 500.0) ** (-exponent), np.ones(w.size)])


def fit_scatter(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    exponent: float = DEFAULT_EXPONENT,
    window: tuple[float, float] = DEFAULT_FIT_WINDOW,
) -> ScatterModel:
    """Fit the scattering background on the non-absorbing window.

    Parameters
    ----------
    wavelength_nm
        Wavelength axis, shape ``(n_wavelength,)``.
    absorbance
        Spectra, shape ``(n_wavelength,)`` or ``(n_wavelength, n_spectra)``.
    exponent
        Fixed power-law exponent.
    window
        ``(low, high)`` wavelength limits of the fit region, in nm.

    Returns
    -------
    ScatterModel

    Raises
    ------
    ValueError
        If fewer than three points fall inside ``window``.
    """
    w = np.asarray(wavelength_nm, dtype=float)
    Y = np.asarray(absorbance, dtype=float)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]

    mask = (w >= window[0]) & (w <= window[1])
    if mask.sum() < 3:
        raise ValueError(
            f"scatter fit window {window} contains {mask.sum()} points; need >= 3"
        )

    basis = _basis(w, exponent)
    coef, *_ = np.linalg.lstsq(basis[mask], Y[mask], rcond=None)
    return ScatterModel(
        exponent=float(exponent),
        amplitude=coef[0],
        offset=coef[1],
        window=window,
    )


def remove_scatter(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    exponent: float = DEFAULT_EXPONENT,
    window: tuple[float, float] = DEFAULT_FIT_WINDOW,
    return_model: bool = False,
):
    """Subtract the fitted scattering background.

    Parameters
    ----------
    wavelength_nm, absorbance, exponent, window
        As for :func:`fit_scatter`.
    return_model
        If true, also return the :class:`ScatterModel`.

    Returns
    -------
    ndarray
        Corrected spectra, same shape as ``absorbance``.
    ScatterModel
        Only when ``return_model`` is true.
    """
    Y = np.asarray(absorbance, dtype=float)
    squeeze = Y.ndim == 1
    model = fit_scatter(wavelength_nm, Y, exponent=exponent, window=window)
    corrected = (Y[:, None] if squeeze else Y) - model.evaluate(wavelength_nm)
    if squeeze:
        corrected = corrected[:, 0]
    return (corrected, model) if return_model else corrected


def exponent_sensitivity(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    statistic,
    exponents: tuple[float, ...] = SENSITIVITY_EXPONENTS,
    window: tuple[float, float] = DEFAULT_FIT_WINDOW,
) -> dict[float, np.ndarray]:
    """Evaluate a statistic across a range of scatter exponents.

    Use this to attach a systematic uncertainty to any quantity derived from
    scatter-corrected spectra.

    Parameters
    ----------
    wavelength_nm, absorbance, window
        As for :func:`fit_scatter`.
    statistic
        Callable ``f(wavelength_nm, corrected) -> value``.
    exponents
        Exponents to span.

    Returns
    -------
    dict
        Mapping exponent to the value of ``statistic``.

    Examples
    --------
    >>> band = lambda w, a: np.trapezoid(a[(w >= 395) & (w <= 430)],
    ...                                  w[(w >= 395) & (w <= 430)])
    >>> vals = exponent_sensitivity(wl, delta_a, band)  # doctest: +SKIP
    >>> systematic = max(vals.values()) - min(vals.values())  # doctest: +SKIP
    """
    out = {}
    for n in exponents:
        corrected = remove_scatter(wavelength_nm, absorbance, exponent=n, window=window)
        out[float(n)] = statistic(np.asarray(wavelength_nm, dtype=float), corrected)
    return out
