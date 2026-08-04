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

__all__ = ["BandSeries", "integrate_band", "band_series", "bootstrap_ci",
           "DIAGNOSTIC_NM", "diagnostic_trace", "fit_single_exponential",
           "table_diagnostic_wavelengths", "table_exponential_fits",
           "table_dose_budget"]


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


# ---------------------------------------------------------------------------
# Diagnostic-wavelength route
# ---------------------------------------------------------------------------
# The standard presentation in the microspectrophotometry literature (Weik et
# al. 2002; Sutton et al. 2013) reads absorbance at a small number of FIXED
# wavelengths, chosen from published assignments, and plots them against
# absorbed dose.  No band centre or width is a free parameter, so nothing here
# depends on fitting into the noisy blue flank of the spectrum.

DIAGNOSTIC_NM = (310.0, 328.0, 400.0, 412.0, 480.0, 580.0)
"""Fixed wavelengths, nm.

310  the positive band that grows in both soak states (protein chemistry)
328  reported absorbance of the protein--S--S--TNB mixed disulphide
400  disulphide radical anion, after Sutton et al. (2013)
412  free TNB(2-) thiolate in solution; the label band
480  isosbestic point reported by Sutton et al. (2013); used here as a check
580  solvated electron, after Sutton et al. (2013)
"""

# Bound on the fitted characteristic dose.  A trace that has not begun to
# saturate within the measured range will run to this bound; that is
# informative (it means "no saturation seen") and must be reported as a bound
# rather than as a fitted value, so the bound is deliberately far outside any
# dose reached here.
D1_BOUND_MGY = 200.0
# Minimum fraction of the modelled decay that must fall inside the measured
# dose range for d1 (and any D90 derived from it) to count as determined.
DECAY_OBSERVED_MIN = 0.5


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
    out = {"d1_MGy": np.nan, "D90_MGy": np.nan, "R2_single": np.nan,
           "decay_observed_frac": np.nan,
           "dA_total": np.nan, "d1_at_bound": False, "D90_is_lower_bound": False,
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
    return out


def table_diagnostic_wavelengths(results, pairs, half_width=4.0):
    """DTNB-minus-apo difference at each diagnostic wavelength, per pair.

    The difference is taken on a common dose grid running to the smaller of the
    two conditions' analysed maxima, so neither arm contributes beyond where the
    other stops.
    """
    import numpy as np
    import pandas as pd

    rows = []
    for dk, ak in pairs:
        if dk not in results or ak not in results:
            continue
        T = float(results[dk].condition.transmission_pct)
        dmax = min(results[dk].max_valid_dose_MGy(),
                   results[ak].max_valid_dose_MGy())
        for lam in DIAGNOSTIC_NM:
            Dd, yd = diagnostic_trace(results[dk], lam, half_width)
            Da, ya = diagnostic_trace(results[ak], lam, half_width)
            if Dd.size < 8 or Da.size < 8:
                continue
            grid = np.linspace(0.0, dmax, 400)
            diff = np.interp(grid, Dd, yd) - np.interp(grid, Da, ya)
            low = grid <= min(2.0, dmax)
            slope = (np.polyfit(grid[low], diff[low], 1)[0]
                     if low.sum() > 3 else np.nan)
            rows.append({
                "transmission_pct": T,
                "wavelength_nm": lam,
                "common_dose_max_MGy": round(float(dmax), 3),
                "dA_diff_end": round(float(diff[-1]), 4),
                "initial_slope_per_MGy": round(float(slope), 5),
            })
    return pd.DataFrame(rows)


def table_exponential_fits(results, pairs=None, half_width=4.0):
    """Single-exponential fit and D90 at each diagnostic wavelength."""
    import pandas as pd

    import numpy as np

    rows = []
    for key, r in results.items():
        for lam in DIAGNOSTIC_NM:
            D, y = diagnostic_trace(r, lam, half_width)
            if D.size < 10:
                continue
            # Single-arm traces keep the falling-segment restriction; see
            # fit_single_exponential for why the difference does not need it.
            f = fit_single_exponential(D, y)
            rows.append({
                "arm": "single",
                "condition": key,
                "wavelength_nm": lam,
                "dose_max_MGy": round(float(D[-1]), 3),
                "n_frames_fitted": f["n_frames_fitted"],
                "dA_extremum": round(f["dA_extremum"], 4),
                "dose_at_extremum_MGy": round(f["dose_at_extremum_MGy"], 3),
                "recovery_frac": round(f["recovery_frac"], 3),
                "d1_MGy": round(f["d1_MGy"], 3),
                "d1_at_bound": f["d1_at_bound"],
                "D90_MGy": round(f["D90_MGy"], 3),
                "D90_is_lower_bound": f["D90_is_lower_bound"],
                "R2_single": round(f["R2_single"], 4),
                "decay_observed_frac": round(f["decay_observed_frac"], 3),
            })

    # The label-specific quantity: DTNB minus its matched apo control, on a
    # common dose grid.  These are the values the figure and the text quote.
    for dk, ak in pairs or ():
        if dk not in results or ak not in results:
            continue
        for lam in DIAGNOSTIC_NM:
            Dd, yd = diagnostic_trace(results[dk], lam, half_width)
            Da, ya = diagnostic_trace(results[ak], lam, half_width)
            if Dd.size < 10 or Da.size < 10:
                continue
            dmax = min(results[dk].max_valid_dose_MGy(),
                       results[ak].max_valid_dose_MGy())
            grid = np.linspace(0.0, float(dmax), 600)
            diff = np.interp(grid, Dd, yd) - np.interp(grid, Da, ya)
            f = fit_single_exponential(grid, diff,
                                       falling_segment_only=False)
            rows.append({
                "arm": "difference",
                "condition": f"{dk} - {ak}",
                "wavelength_nm": lam,
                "dose_max_MGy": round(float(dmax), 3),
                "n_frames_fitted": f["n_frames_fitted"],
                "dA_extremum": round(f["dA_extremum"], 4),
                "dose_at_extremum_MGy": round(f["dose_at_extremum_MGy"], 3),
                "recovery_frac": round(f["recovery_frac"], 3),
                "d1_MGy": round(f["d1_MGy"], 3),
                "d1_at_bound": f["d1_at_bound"],
                "D90_MGy": round(f["D90_MGy"], 3),
                "D90_is_lower_bound": f["D90_is_lower_bound"],
                "R2_single": round(f["R2_single"], 4),
                "decay_observed_frac": round(f["decay_observed_frac"], 3),
            })
    return pd.DataFrame(rows)


# Published room-temperature diffraction lifetimes, for scale.  Each is a
# measured D(1/2) or equivalent from Owen et al. (2012), spanning both protein
# and collection speed; the spread is the point, since it bounds how far the
# feasibility argument can be pushed.
RT_DIFFRACTION_LIFETIMES_MGY = (
    ("BEV2 (Owen 2012, 1 Hz)", 0.097),
    ("A2AAR (Owen 2012, 1 Hz)", 0.108),
    ("FcgRIIIa stop-start (Owen 2012)", 0.186),
    ("A2AAR (Owen 2012, 25 Hz)", 0.192),
    ("FcgRIIIa continuous <11 Hz (Owen 2012)", 0.257),
    ("FcgRIIIa 25 Hz (Owen 2012)", 0.373),
)


def table_dose_budget(results, pairs, tnb_window=(408.0, 416.0),
                      noise_floor=0.004, half_width=4.0):
    """Dose needed to see the label leave, against published crystal lifetimes.

    ``noise_floor`` is the measured baseline floor on the difference, not a
    nominal value: the internal-zero window is not exactly zero in DTNB films.
    """
    import numpy as np
    import pandas as pd

    lam = 0.5 * (tnb_window[0] + tnb_window[1])
    rows = []
    for dk, ak in pairs:
        if dk not in results or ak not in results:
            continue
        dmax = min(results[dk].max_valid_dose_MGy(),
                   results[ak].max_valid_dose_MGy())
        Dd, yd = diagnostic_trace(results[dk], lam, half_width)
        Da, ya = diagnostic_trace(results[ak], lam, half_width)
        if Dd.size < 8 or Da.size < 8:
            continue
        grid = np.linspace(0.0, dmax, 800)
        diff = np.interp(grid, Dd, yd) - np.interp(grid, Da, ya)
        diff = diff - diff[0]
        # Two defensible definitions of "detectable", reported side by side
        # because they differ by an order of magnitude and the difference is
        # a property of the data, not of the analysis:
        #
        #  empirical  the dose at which the measured difference first exceeds
        #             the floor AND stays above it.  Persistence is required so
        #             that a single noisy frame cannot set the answer.
        #  linear     the floor divided by the initial slope over the first
        #             1 MGy.  This is what a linear extrapolation from the
        #             early response would predict.
        #
        # The measured curve is strongly convex: most of the label loss happens
        # in the first few tenths of a MGy and then flattens, so the empirical
        # dose is much the smaller of the two.  The linear figure is the
        # conservative one and is what a proposal should be costed against.
        run = max(3, int(0.01 * grid.size))
        above = np.abs(diff) >= noise_floor
        dose_clear = np.nan
        for i in range(above.size - run):
            if above[i:].all():
                dose_clear = float(grid[i])
                break
        low = grid <= min(1.0, dmax)
        slope = (float(np.polyfit(grid[low], diff[low], 1)[0])
                 if low.sum() > 3 else np.nan)
        dose_linear = (abs(noise_floor / slope)
                       if np.isfinite(slope) and slope != 0 else np.nan)
        rate = float(results[dk].condition.transmission_pct) / 100.0 * 1.0418
        for name, life in RT_DIFFRACTION_LIFETIMES_MGY:
            rows.append({
                "condition": dk,
                "dose_rate_MGy_per_s": round(rate, 3),
                "dose_to_clear_noise_MGy": round(dose_clear, 3),
                "dose_linear_extrapolation_MGy": round(dose_linear, 3),
                "initial_slope_per_MGy": round(slope, 5),
                "noise_floor_dA": noise_floor,
                "diffraction_limit_source": name,
                "diffraction_lifetime_MGy": life,
                "detection_dose_over_lifetime": round(dose_clear / life, 2),
                "linear_dose_over_lifetime": round(dose_linear / life, 2),
                "crystals_to_average": int(np.ceil((dose_linear / life) ** 2))
                if np.isfinite(dose_linear) and dose_linear > life else 1,
            })
    return pd.DataFrame(rows)
