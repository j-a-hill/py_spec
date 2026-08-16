"""Minimal quantitation module for the formal figure set (figures 1-4).

Trimmed from crystalspec.quantify: only the three functions figures 1-4
actually call (diagnostic_trace, fit_single_exponential, integrate_band)
and the constants they need. The full quantify.py (tables 1/3/4/8/9/10,
bootstrap CIs, dose-budget) is not reproduced here because none of it
feeds figures 1-4; see the crystalspec package itself for the complete
analysis.
"""

from __future__ import annotations

import numpy as np

# A d1 fit that reaches this bound signals "no saturation seen within the
# measured range" and must be reported as a bound, not a fitted value.
D1_BOUND_MGY = 200.0

# Minimum fraction of the modelled decay that must fall inside the measured
# dose range for d1 (and any D50/D90 derived from it) to count as determined.
DECAY_OBSERVED_MIN = 0.5

DIAGNOSTIC_NM = (310.0, 328.0, 400.0, 412.0, 480.0, 580.0)


def diagnostic_trace(result, wavelength_nm, half_width=4.0,
                     min_positions=3):
    """Mean difference absorbance in a narrow window, against dose.

    Returns ``(dose_MGy, delta_a)`` over the analysed frames only.  The
    contributing set of film positions is held fixed at the ``min_positions``
    positions surviving longest, so that a position dropping out part way
    through cannot introduce a step into the mean.
    """
    import numpy as np

    r = result
    valid = r.valid() & (r.dose_MGy <= r.max_valid_dose_MGy())
    if valid.sum() < 2:
        return np.array([]), np.array([])
    finite = np.isfinite(r.delta_a[:, 0, :])
    last = np.array([np.nonzero(row)[0][-1] if row.any() else -1
                     for row in finite])
    keep = np.argsort(last)[::-1][:min_positions]
    band = ((r.wavelength_nm >= wavelength_nm - half_width)
            & (r.wavelength_nm <= wavelength_nm + half_width))
    with np.errstate(invalid="ignore"):
        per_pos = np.nanmean(r.delta_a[keep][:, band, :][:, :, valid], axis=1)
        y = np.nanmean(per_pos, axis=0)
    return r.dose_MGy[valid], y


def fit_single_exponential(dose_MGy, delta_a, falling_segment_only=True):
    """Fit ``A = A0 + B exp(-D / d1)`` and report D90.

    The functional form follows Sutton et al. (2013).

    ON ``falling_segment_only``, and why it is no longer needed for the label.
    The 412 nm trace of the DTNB arm ALONE is not monotonic: it falls to an
    extremum within the first few MGy and then returns partway, by 11 % of the
    fall at 5 % transmission and by 76 % at 25 %.  That recovery is not the
    label coming back and it is not a spectrometer reset -- DTNB_25 records no
    resets at all.  It is the protein growth band centred near 310 nm, whose
    red tail reaches into the 412 nm window and lifts the trace as it builds:
    over the recovering segment the 412 nm rise correlates with the 310 nm
    growth at r = +0.96, while the 470-500 nm internal zero stays flat.

    Subtracting the matched apo control removes it, because the growth is
    common to both arms.  On the DTNB-minus-apo difference the recovery is
    0-5 % and the trace is monotonic, so the difference is fitted over its
    full range.  ``falling_segment_only`` remains available for single-arm
    traces, where the shape still requires it.

    ``D90`` is the dose at which the fitted curve has completed 90 % of the
    excursion it makes over the fitted segment.  ``D90_is_lower_bound`` marks
    the case where that dose sits within the last tenth of the fitted range,
    meaning the segment had not flattened before it ended.  Returns a dict with
    ``NaN`` entries when the fit does not converge.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    D = np.asarray(dose_MGy, dtype=float)
    y = np.asarray(delta_a, dtype=float)
    good = np.isfinite(D) & np.isfinite(y)
    out = {"d1_MGy": np.nan, "D90_MGy": np.nan, "D50_MGy": np.nan,
           "R2_single": np.nan,
           "decay_observed_frac": np.nan,
           "dA_total": np.nan, "d1_at_bound": False, "D90_is_lower_bound": False,
           "D50_is_lower_bound": False,
           "dA_extremum": np.nan, "dose_at_extremum_MGy": np.nan,
           "recovery_frac": np.nan, "n_frames_fitted": 0,
           "n_frames": int(good.sum())}
    if good.sum() < 10:
        return out
    D, y = D[good], y[good]
    out["dA_total"] = float(y[-1] - y[0])
    out["dA_extremum"] = float(y[np.argmax(np.abs(y - y[0]))] - y[0])
    out["dose_at_extremum_MGy"] = float(D[np.argmax(np.abs(y - y[0]))])
    out["recovery_frac"] = (
        float((y[-1] - y[np.argmax(np.abs(y - y[0]))]) / out["dA_extremum"])
        if abs(out["dA_extremum"]) > 1e-12 else np.nan)
    if falling_segment_only:
        cut = int(np.argmax(np.abs(y - y[0]))) + 1
        if cut >= 10:
            D, y = D[:cut], y[:cut]
    out["n_frames_fitted"] = int(D.size)

    def model(x, a0, b, d1):
        return a0 + b * np.exp(-x / d1)

    span = float(np.ptp(y)) or 1.0
    try:
        popt, _ = curve_fit(
            model, D, y, p0=[y[-1], y[0] - y[-1], max(D.max() / 3, 1e-2)],
            bounds=([y.min() - 5 * span, -5 * span, 1e-3],
                    [y.max() + 5 * span, 5 * span, D1_BOUND_MGY]),
            maxfev=60000)
    except (RuntimeError, ValueError):
        return out
    pred = model(D, *popt)
    ss, st = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
    out["R2_single"] = float(1.0 - ss / st) if st > 0 else np.nan
    d1 = float(popt[2])
    out["d1_MGy"] = d1
    out["d1_at_bound"] = bool(d1 >= 0.999 * D1_BOUND_MGY)
    # D90 is defined EMPIRICALLY over the measured range: the dose at which the
    # fitted curve first attains 90 % of the total excursion it makes across
    # the doses actually collected.  The alternative, ln(10) * d1, is the dose
    # to 90 % of the asymptotic excursion, which for a trace that has not
    # saturated lies far outside the measured range and is not interpretable.
    # Defined this way, a D90 close to dose_max means "had not saturated by the
    # end of the measurement" and is a LOWER BOUND set by where the data stop.
    total = pred[-1] - pred[0]
    if abs(total) < 1e-12:
        return out
    reached = np.nonzero(np.abs(pred - pred[0]) >= 0.9 * abs(total))[0]
    out["D90_MGy"] = float(D[reached[0]]) if reached.size else float(D[-1])

    # D50 (half-dose), by the SAME empirical, measured-range definition as
    # D90 above -- the dose at which the fitted curve first attains 50 % of
    # its measured excursion, not D90 / 2.  For a single exponential the two
    # are related by a FIXED ratio, D50 = D90 * ln(2)/ln(10) = D90 * 0.301,
    # because the curve is exponential rather than linear in dose; halving
    # D90 would overstate D50 by roughly 3x on this data.  D50 is computed
    # from the same measured-range crossing (not the ratio) so it inherits
    # exactly the same lower-bound behaviour as D90 when the fit has not
    # observed enough of the curve.
    reached50 = np.nonzero(np.abs(pred - pred[0]) >= 0.5 * abs(total))[0]
    out["D50_MGy"] = float(D[reached50[0]]) if reached50.size else float(D[-1])

    # How much of the decay the model postulates was actually traversed by the
    # measurement.  This is the diagnostic that matters, and it is not visible
    # in R^2: a trace can be fitted with R^2 near 0.99 by the early, nearly
    # linear part of an exponential whose time constant is far outside the
    # measured range, in which case d1 is unidentifiable and every quantity
    # derived from it is an extrapolation.  At 100 % transmission the fit
    # observes 0.14 of its own modelled decay and d1 lands anywhere between
    # 0.16 and 200 MGy across position bootstrap resamples; at the other three
    # settings the figure is 0.996 or better.
    out["decay_observed_frac"] = float(1.0 - np.exp(-float(D[-1]) / d1))
    out["D90_is_lower_bound"] = bool(
        out["D90_MGy"] >= 0.9 * float(D[-1])
        or out["decay_observed_frac"] < DECAY_OBSERVED_MIN)
    # D50 needs less of the curve to be identified than D90 does (it is
    # reached earlier), but it still inherits the same overall
    # non-identifiability when the fit itself is unconstrained.
    out["D50_is_lower_bound"] = bool(
        out["D50_MGy"] >= 0.9 * float(D[-1])
        or out["decay_observed_frac"] < DECAY_OBSERVED_MIN)
    return out


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
