"""Minimal figure module for the formal deliverable set (figures 1-4).

Trimmed from crystalspec.figures: only the four functions that produce
figures 1-4 (the main-text spectroscopy results, per the formal-figures
scope decision). The other 12 figure functions in the full crystalspec
package (viability/feasibility figures, QC, buffer controls, and earlier
superseded presentations) are not reproduced here -- see the crystalspec
package itself for the complete analysis and its full figure set.

Canvas conventions per the formal-figures skill: no on-canvas titles,
frameless legends, no interpretive/directional annotations on reference
lines (their meaning is stated in the external caption file instead).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from crystalspec.scatter import remove_scatter
from crystalspec.style import (
    SOAK, SOAK_LABEL, TRANSMISSION, band_mask, check_overlaps, panel_label,
    shade_band, twin_time_axis,
)

from quantify_min import diagnostic_trace, fit_single_exponential, integrate_band

# Wavelength -> assignment label, for figure_diagnostic_wavelengths panel B.
# NOTE this is a different object from quantify_min's DIAGNOSTIC_NM (a tuple
# of the same six wavelengths, used there for table iteration).
DIAGNOSTIC_NM = {310.0: "growth band",
                 328.0: "protein\u2013S\u2013S\u2013TNB",
                 400.0: "disulphide radical",
                 412.0: "free TNB$^{2-}$",
                 480.0: "isosbestic",
                 580.0: "solvated e$^-$"}

RT_LIFETIME_BY_DOSE_RATE = {0.261: 0.257, 0.521: 0.300, 1.042: 0.373}


def figure_original_idiom(results, out_path, doses_MGy=(0.0, 0.5, 2.0, 5.0, 20.0),
                          wavelengths_nm=(350.0, 412.0), half_width=4.0,
                          wl_range=(295.0, 700.0),
                          condition="DTNB_25", control="apo_25", xaxis="dose"):
    """Figure 10: the two original plot types, corrected.

    The original analysis produced two figures per condition: spectra at
    intervals through the acquisition, and absorbance at chosen wavelengths
    against time.  Both were the right choice and both are reproduced here.  The
    changes are corrections, not restyling:

    * The wavelength axis starts at 290 nm rather than 179 nm.  Below that the
      spectrometer delivers no usable light -- the scatter across wavelength is
      several-fold the real noise -- so the original figure's left third was
      instrument noise plotted at full amplitude.
    * Spectrometer resets are repaired rather than drawn.  The original
      wavelength-against-time traces contain abrupt drops of order 0.1 OD where
      the detector rezeroed, and the apparent recovery after each was read as
      signal.
    * The independent variable is absorbed dose rather than elapsed time.  Dose
      is the quantity the chemistry responds to and the only axis on which
      conditions collected at different transmissions can be compared; elapsed
      time additionally depends on when the shutter was opened, which was manual
      and is only known to within a second or so.  Dose is computed from the
      detected shutter opening onwards, so the flat leading section of the
      original traces is gone.
    * The matched apo control is drawn on the same axes.  Without it the growth
      near 310-350 nm cannot be attributed to the label, because it occurs in
      unlabelled films too.
    * Frames after the film becomes optically compromised are dropped, so late
      traces show chemistry rather than the sample turning opaque.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    doses_MGy
        Absorbed doses at which to draw spectra, MGy.
    wavelengths_nm
        Wavelengths to follow against time.
    half_width
        Averaging half-width about each wavelength, nm.
    wl_range
        Wavelength limits for the spectra panels.
    condition, control
        Condition keys for the labelled and matched unlabelled arms.
    xaxis
        ``"dose"`` (default) or ``"time"``.  Dose is preferred -- it is what the
        chemistry responds to, and the only axis on which conditions collected at
        different transmissions are comparable.  A time axis is provided because
        the two are not interchangeable here: the loss superimposes better on
        elapsed time than on dose across a twentyfold range of dose rate, which
        is the inverse dose-rate signature, so being able to draw both is part of
        the evidence rather than a formatting preference.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    wl_lo, wl_hi = wl_range

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.52, wspace=0.30)

    def prep(key, min_positions=3):
        """Valid frames, with the contributing position set held fixed.

        Positions drop out at different times.  Averaging over whatever remains
        makes the mean step whenever one leaves, which is an artefact of
        changing composition rather than a change in the sample.  The set is
        therefore fixed at the positions surviving longest.
        """
        r = results[key]
        v = r.valid() & (r.dose_MGy <= r.max_valid_dose_MGy())
        finite = np.isfinite(r.delta_a[:, r.delta_a.shape[1] // 2, :])
        last = np.array([np.nonzero(row)[0][-1] if row.any() else -1
                         for row in finite])
        keep_pos = np.argsort(last)[::-1][:max(min_positions, 1)]
        cut = int(last[keep_pos].min()) + 1
        v = v & (np.arange(v.size) < cut)
        return r, v, keep_pos

    # --- (a, b) spectra at intervals, both arms ---
    for ax, key, colour in ((ax_a, condition, SOAK["DTNB"]),
                            (ax_b, control, SOAK["apo"])):
        r, v, pos = prep(key)
        d = (r.dose_MGy[v] if xaxis == "dose"
             else r.time_s[v] - r.time_s[v][0])
        wl = r.wavelength_nm
        keep = (wl >= wl_lo) & (wl <= wl_hi)
        samples = (doses_MGy if xaxis == "dose"
                   else tuple(np.round(np.asarray(doses_MGy)
                                       / max(r.dose_MGy[v][-1], 1e-9)
                                       * (r.time_s[v][-1] - r.time_s[v][0]), 1)))
        shades = np.linspace(0.25, 1.0, len(samples))
        for dd, sh in zip(samples, shades):
            if dd > d.max():
                continue
            i = int(np.argmin(abs(d - dd)))
            with np.errstate(invalid="ignore"):
                y = np.nanmean(r.delta_a[pos][:, keep, :][:, :, i], axis=0)
            # A small legend rather than in-axes labels: the DTNB traces cross
            # near 365 nm, so no single wavelength orders them for direct
            # labelling and any fixed anchor produces collisions.
            ax.plot(wl[keep], y, color=colour, lw=1.0, alpha=sh,
                    label=(f"{dd:g} MGy" if xaxis == "dose" else f"{dd:g} s"))
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("\u0394A")
        ax.annotate(SOAK_LABEL.get(key.split("_")[0], key.split("_")[0]),
                    xy=(0.97, 0.94), xycoords="axes fraction", ha="right",
                    va="top", fontsize=5.6, color=colour)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=4.6,
                  framealpha=0.0, handlelength=1.2, labelspacing=0.22,
                  title=("absorbed dose" if xaxis == "dose" else "elapsed time"),
                  title_fontsize=4.6)

    # --- (c, d) chosen wavelengths against time, both arms ---
    for ax, lam in zip((ax_c, ax_d), wavelengths_nm):
        for key, colour in ((condition, SOAK["DTNB"]), (control, SOAK["apo"])):
            r, v, pos = prep(key)
            bw = ((r.wavelength_nm >= lam - half_width)
                  & (r.wavelength_nm <= lam + half_width))
            x = (r.dose_MGy[v] if xaxis == "dose"
                 else r.time_s[v] - r.time_s[v][0])
            with np.errstate(invalid="ignore"):
                band = np.nanmean(r.delta_a[pos][:, bw, :][:, :, v], axis=1)
                y = np.nanmean(band, axis=0)
                sd = np.nanstd(band, axis=0)
            step = max(1, int(x.size / 200))
            ax.plot(x[::step], y[::step], color=colour, lw=1.2,
                    label=SOAK_LABEL.get(key.split("_")[0], key.split("_")[0]))
            ax.fill_between(x[::step], (y - sd)[::step], (y + sd)[::step],
                            color=colour, alpha=0.16, lw=0)
        ax.axhline(0, color="0.6", lw=0.5)
        # A per-panel key rather than labels anchored to the end of each trace.
        # The traces do not all end at the same x, so a fixed offset from the
        # last point put the apo label across the neighbouring y-axis title.
        ax.legend(loc="upper left", fontsize=5.2, framealpha=0.0,
                  handlelength=1.3, labelspacing=0.25, borderpad=0.25)
        ax.set_xlabel("Absorbed dose (MGy)" if xaxis == "dose"
                      else "Time since shutter opening (s)")
        ax.set_ylabel(f"\u0394A at {lam:.0f} nm")
        if xaxis == "dose":
            # Both arms in these panels are one transmission, so a single dose
            # rate maps the axis onto seconds.  The chapter figures carry the
            # time axis; keeping it means the two can be read against each other.
            rr = results[condition]
            vv = rr.valid() & (rr.dose_MGy <= rr.max_valid_dose_MGy())
            twin_time_axis(ax, rr.dose_MGy[vv],
                           rr.time_s[vv] - rr.time_s[vv][0])

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_diagnostic_wavelengths(results, out_path, half_width=4.0,
                                  fit_dose_max=2.0, diagnostics=None):
    """Figure 7: the standard microspectrophotometry presentation.

    This follows the convention established for online microspectrophotometry
    of irradiated crystals (Weik et al., 2002; Sutton et al., 2013): absorbance
    change is read at FIXED diagnostic wavelengths assigned from the
    literature, plotted against absorbed dose, and fitted with an exponential
    of the form A = A0 + B exp(-D / d1).  No band centre is a free parameter,
    which is what makes the presentation robust -- the alternative, fitting
    band positions, is not supported by these spectra because the blue flank of
    the growth band lies in the throughput-dead deep UV.

    Panel (a) shows the difference spectrum with the two wavelengths the argument
    turns on marked, so the reader can see what is being sampled.  The remaining
    diagnostic positions are listed in the panel (b) legend rather than drawn:
    six guide lines on a small axes obscured the spectra they annotate.  The
    fitted exponentials are reported in table 13 rather than overlaid, since at
    this size the fit and the data were indistinguishable.  Panels (b) and (c)
    give the dose response at those wavelengths for a DTNB film and its matched
    apo control.  Panel (d) gives the label-specific quantity: DTNB minus apo
    at each wavelength, at the common dose of each pair.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    half_width
        Averaging half-width about each diagnostic wavelength, nm.
    fit_dose_max
        Upper dose for the initial-slope fit in panel (d), MGy.
    diagnostics
        Mapping of wavelength to label; defaults to ``DIAGNOSTIC_NM``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.optimize import curve_fit

    out_path = Path(out_path)
    diag = dict(diagnostics or DIAGNOSTIC_NM)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    def valid_idx(key):
        r = results[key]
        return r.valid() & (r.dose_MGy <= r.max_valid_dose_MGy())

    def trace(key, lam):
        # Route through diagnostic_trace so the contributing set of film
        # positions is held FIXED.  Averaging over whatever positions remain
        # valid at each frame puts a step in the mean wherever one drops out;
        # that is what produced the apparent dip near 10.7 MGy in the earlier
        # version of this figure, exactly where DTNB_25 position 3 ends.
        return diagnostic_trace(results[key], lam, half_width)

    def diff_trace(dk, ak, lam):
        """DTNB minus matched apo, on a common dose grid.

        The subtraction matters at 412 nm specifically.  The protein growth
        band centred near 310 nm has a red tail that reaches into the 412 nm
        window, so the DTNB trace alone falls to an extremum and then appears
        to recover as that tail builds up underneath it; the apparent recovery
        correlates with the 310 nm growth at r = +0.96 while the 470-500 nm
        null stays flat, and it cancels in the difference.
        """
        Dd, yd = trace(dk, lam)
        Da, ya = trace(ak, lam)
        if Dd.size < 8 or Da.size < 8:
            return np.array([]), np.array([])
        dmax = min(results[dk].max_valid_dose_MGy(),
                   results[ak].max_valid_dose_MGy())
        grid = np.linspace(0.0, dmax, 600)
        return grid, np.interp(grid, Dd, yd) - np.interp(grid, Da, ya)

    def expo(D, A0, B, d1):
        return A0 + B * np.exp(-D / d1)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.42, bottom=0.34, top=0.80)

    greys = plt.cm.viridis(np.linspace(0.1, 0.88, len(diag)))
    lam_colour = {lam: greys[i] for i, lam in enumerate(sorted(diag))}

    # --- (a) where the diagnostics sit on the spectrum ---
    for key, colour, label in (("DTNB_25", SOAK["DTNB"], "DTNB"),
                               ("apo_25", SOAK["apo"], "apo")):
        r = results[key]
        v = valid_idx(key)
        i = int(np.nonzero(v)[0][-1])
        wl = r.wavelength_nm
        keep = (wl >= 285.0) & (wl <= 620.0)
        # Average over the positions surviving longest, not over whatever is
        # still valid at the last frame -- the same fixed-set rule the dose
        # traces use, so panel A and the traces describe the same films.
        fin = np.isfinite(r.delta_a[:, 0, :])
        last = np.array([np.nonzero(row)[0][-1] if row.any() else -1
                         for row in fin])
        sel = np.argsort(last)[::-1][:3]
        with np.errstate(invalid="ignore"):
            spec = np.nanmean(r.delta_a[sel][:, keep, i], axis=0)
        ax_a.plot(wl[keep], spec, color=colour, lw=1.1, label=label)
    ax_a.axhline(0, color="0.6", lw=0.5)
    # Only the two wavelengths the argument turns on are marked.  Marking all
    # six put more dashed lines on the panel than there were spectra.
    for lam in (328.0, 412.0):
        ax_a.axvline(lam, color="0.5", lw=0.7, ls=(0, (2, 2)), zorder=0)
        ax_a.annotate(f"{lam:.0f}", xy=(lam, ax_a.get_ylim()[1]),
                      xytext=(2, -3), textcoords="offset points",
                      fontsize=5.0, color="0.4", va="top")
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("\u0394A")
    ax_a.legend(loc="upper right", fontsize=5.5, framealpha=0.0)

    # --- (b) the label-specific difference, and (c) the rate ---
    #
    # Panels B and C previously showed DTNB and apo raw traces side by side,
    # which asked the reader to do the subtraction by eye; the difference is
    # the quantity every claim rests on, so it is now drawn directly.  The old
    # panel (d), a bar chart of endpoint differences per transmission, is
    # dropped: it compressed each whole dose series into one number and the
    # same information is in table 8.
    fit_note = []
    for lam in sorted(diag):
        g, diff = diff_trace("DTNB_25", "apo_25", lam)
        if g.size < 12:
            continue
        step = max(1, int(g.size / 200))
        ax_b.plot(g[step // 2::step], diff[step // 2::step],
                  color=lam_colour[lam], lw=1.0, label=f"{lam:.0f} nm")
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Absorbed dose (MGy)")
    ax_b.set_ylabel("\u0394A, DTNB \u2212 apo")
    rr = results["DTNB_25"]
    vv = rr.valid() & (rr.dose_MGy <= rr.max_valid_dose_MGy())
    twin_time_axis(ax_b, rr.dose_MGy[vv], rr.time_s[vv] - rr.time_s[vv][0])
    ax_b.legend(loc="lower left", fontsize=5.0, ncol=2, framealpha=0.0,
                handlelength=1.2, labelspacing=0.25, borderpad=0.3)

    # (c) the rate: the label band against dose at every transmission, with the
    # fitted exponential overlaid.  This is the panel the original analysis had
    # and the minimal set had lost -- the D90 values quoted in the text are
    # read off these curves, and without it they arrive unsupported.
    for dk, ak in pairs:
        T = int(results[dk].condition.transmission_pct)
        g, diff = diff_trace(dk, ak, 412.0)
        if g.size < 12:
            continue
        step = max(1, int(g.size / 200))
        fit = fit_single_exponential(g, diff, falling_segment_only=False)
        # Report every converged fit with its R2 rather than hiding the weaker
        # ones: suppressing the 100 % entry (R2 0.75) left the figure silently
        # short of a value the text quotes.  The reader weighs the quality.
        # A fit counts only if the measurement traversed most of the decay it
        # postulates.  R^2 does not test this: at 100 % transmission the single
        # exponential reaches R^2 0.76 while observing 0.14 of its own modelled
        # decay, and d1 then ranges over three orders of magnitude across
        # position-bootstrap resamples.  Such a fit is shown as a lower bound
        # with no curve drawn, rather than as a number the reader would trust.
        ok_fit = (np.isfinite(fit["R2_single"]) and np.isfinite(fit["D90_MGy"])
                  and not fit["D90_is_lower_bound"])
        lab = f"{T}%"
        if ok_fit:
            lab = (f"{T}%, $D_{{50/90}}$ {fit['D50_MGy']:.1f}/"
                   f"{fit['D90_MGy']:.1f} MGy ($R^2$ {fit['R2_single']:.2f})")
        elif np.isfinite(fit["D90_MGy"]):
            lab = (f"{T}%, $D_{{50/90}}$ > {fit['D50_MGy']:.1f}/"
                   f"{fit['D90_MGy']:.1f} MGy (not saturated)")
        ax_c.plot(g[step // 2::step], diff[step // 2::step],
                  color=TRANSMISSION[T], lw=1.0, label=lab)
        if ok_fit:
            try:
                popt, _ = curve_fit(
                    expo, g, diff,
                    p0=[diff[-1], diff[0] - diff[-1], max(g.max() / 3, 1e-2)],
                    bounds=([-1, -1, 1e-3], [1, 1, 200]), maxfev=40000)
                ax_c.plot(g, expo(g, *popt), color=TRANSMISSION[T], lw=0.7,
                          ls=(0, (3, 2)), alpha=0.8, zorder=5)
                fit_note.append((dk, fit["D90_MGy"], fit["R2_single"]))
            except (RuntimeError, ValueError):
                pass
    ax_c.axhline(0, color="0.6", lw=0.5)
    # Linear, truncated to the dose every condition actually reaches.  A log
    # axis spanning 0.003-23 MGy gives half the width to the first few frames,
    # where the trace is noisiest and nothing has happened yet.
    ax_c.set_xlim(0.0, 5.0)
    ax_c.set_xlabel("Absorbed dose (MGy)")
    ax_c.set_ylabel("\u0394A at 412 nm, DTNB \u2212 apo")
    # Below the axes: at this panel size every in-axes corner is occupied by
    # a trace, and four entries with fitted values are too wide to inset.
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                fontsize=4.4, framealpha=0.0,
                title="transmission (dashed: fitted decay)",
                title_fontsize=4.8, handlelength=1.4, labelspacing=0.22,
                columnspacing=1.0, borderpad=0.2)

    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_experiment_design(results, out_path, tnb_window=(408.0, 416.0),
                             slope_dose_max=1.0, noise_floor=0.004,
                             lifetime_by_dose_rate=None,
                             reference_lifetime_MGy=0.257,
                             min_ground_state=0.05):
    """Figure 9: what a joint diffraction and spectroscopy experiment needs.

    Two panels, one argument each, with the minimum of reference marks needed to
    read them.  An earlier version carried shaded lifetime bands, a shaded noise
    floor and a separate bar panel; with four data traces in a small axes those
    marks occupied more of the figure than the data and are omitted here.  A
    single lifetime line replaces the band, and the detection threshold is not
    drawn at all -- it is a property of the analysis, reported in the text and
    used to compute panel (b), not a feature of the spectra.

    Panel (a): the label-specific absorbance change against dose, over the dose
    a room-temperature crystal survives.  The label is barely touched.

    Panel (b): the dose needed to detect that change, divided by the crystal
    lifetime reported at a comparable collection speed.  Points above one are
    infeasible on a single crystal.  Collecting faster moves the wrong way; the
    overlaid series shows where averaging N crystals puts the same measurement.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    tnb_window
        Label band integration window, nm.
    slope_dose_max
        Upper dose for the initial-rate fit, MGy.
    noise_floor
        Single-measurement detection threshold in absorbance, from the
        internal-zero window.  Not drawn; sets the detection dose in panel (b).
    lifetime_by_dose_rate
        Mapping of dose rate to the published lifetime measured at a comparable
        collection speed, MGy.  Panel (b) needs this rather than a single value:
        the inverse dose-rate effect makes the lifetime itself depend on
        collection speed, and one fixed denominator would hide the trade-off the
        panel exists to show.  Defaults to ``RT_LIFETIME_BY_DOSE_RATE``.
    reference_lifetime_MGy
        Single lifetime drawn as a reference line in panel (a).  The published
        values span 0.097-0.373 MGy; one representative value is marked rather
        than the whole spread, which as a shaded band dominated the panel.
    min_ground_state
        Minimum ground-state label absorbance for fractional quantities.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    lo, hi = tnb_window
    life_map = dict(lifetime_by_dose_rate or RT_LIFETIME_BY_DOSE_RATE)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    def lifetime_at(rate):
        return life_map[min(life_map, key=lambda r: abs(r - rate))]

    def band_trace(key):
        r = results[key]
        v = r.valid() & (r.dose_MGy <= r.max_valid_dose_MGy())
        bw = (r.wavelength_nm >= lo) & (r.wavelength_nm <= hi)
        with np.errstate(invalid="ignore"):
            y = np.nanmean(np.nanmean(r.delta_a[:, bw, :][:, :, v], axis=1), axis=0)
        return r.dose_MGy[v], y

    def ground_state(dk, ak):
        rd, ra = results[dk], results[ak]
        bw = (rd.wavelength_nm >= lo) & (rd.wavelength_nm <= hi)
        return float(np.nanmean([np.nanmean(t.raw_dark[bw]) for t in rd.traces])
                     - np.nanmean([np.nanmean(t.raw_dark[bw]) for t in ra.traces]))

    rows = []
    for dk, ak in pairs:
        Dd, yd = band_trace(dk)
        Da, ya = band_trace(ak)
        dmx = min(Dd.max(), Da.max())
        grid = np.linspace(0, min(slope_dose_max, dmx), 80)
        diff = np.interp(grid, Dd, yd) - np.interp(grid, Da, ya)
        rows.append(dict(
            T=int(results[dk].condition.transmission_pct),
            rate=results[dk].condition.transmission_pct / 100 * 1.042,
            grid=grid, diff=diff - diff[0],
            slope=float(np.polyfit(grid, diff - diff[0], 1)[0]),
            ground=ground_state(dk, ak)))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.34, bottom=0.22, right=0.97)

    # --- (a) the label over a diffraction dataset ---
    xmax = reference_lifetime_MGy * 1.45
    for r in rows:
        show = r["grid"] <= xmax
        ax_a.plot(r["grid"][show], r["diff"][show],
                  color=TRANSMISSION[r["T"]], lw=1.4)
        ax_a.annotate(f"{r['T']}%",
                      xy=(r["grid"][show][-1], r["diff"][show][-1]),
                      xytext=(3, 0), textcoords="offset points",
                      fontsize=5.4, color=TRANSMISSION[r["T"]], va="center")
    ax_a.axvline(reference_lifetime_MGy, color="0.45", lw=0.7, ls=(0, (3, 2)))
    ax_a.annotate("crystal lifetime",
                  xy=(reference_lifetime_MGy, ax_a.get_ylim()[0]),
                  xytext=(-3, 4), textcoords="offset points",
                  fontsize=5.0, color="0.45", ha="right")
    ax_a.set_xlim(0, xmax * 1.12)
    ax_a.set_xlabel("Absorbed dose (MGy)")
    ax_a.set_ylabel(f"\u0394A change, DTNB \u2212 apo\n({lo:.0f}\u2013{hi:.0f} nm)")

    # --- (b) feasibility on one crystal, and the averaging route ---
    usable = [r for r in rows if r["ground"] >= min_ground_state]
    for r in usable:
        ratio = (noise_floor / abs(r["slope"])) / lifetime_at(r["rate"])
        ax_b.plot([r["rate"]], [ratio], "o", color=TRANSMISSION[r["T"]], ms=5.5,
                  zorder=3)
        # Label in dark grey, not the point's own colour: the two lightest
        # transmission shades (5, 10%) are below 4.5:1 text contrast on
        # white and would be illegible at this size.
        ax_b.annotate(f"{r['T']}%", xy=(r["rate"], ratio), xytext=(5, 0),
                      textcoords="offset points", fontsize=5.4,
                      color="0.25", va="center")
    best = min(usable, key=lambda r: noise_floor / abs(r["slope"]))
    best_life = lifetime_at(best["rate"])
    n_avg = np.array([1, 4, 25])  # linear y-axis: a 2x range needs no log scale
    det_n = (noise_floor / np.sqrt(n_avg)) / abs(best["slope"]) / best_life
    ax_b.plot(np.full(n_avg.size, best["rate"]), det_n, color="0.4", lw=0.8,
              marker="s", ms=2.8, zorder=2)
    for N, d in zip(n_avg[1:], det_n[1:]):
        ax_b.annotate(f"average {N}", xy=(best["rate"], d), xytext=(5, 0),
                      textcoords="offset points", fontsize=5.0, color="0.4",
                      va="center")
    # y=1 is the feasibility line (dose to detect / lifetime); left
    # unlabelled on-canvas per the formal-figure convention -- its meaning
    # is stated in the caption instead.
    ax_b.axhline(1.0, color="0.3", lw=0.7)
    # Linear, not log: the plotted dose rates span under a 4x range, and a
    # log axis there produces dense, overlapping minor-tick labels rather
    # than a wider view of anything (house convention: log axes are for
    # >=1-decade ranges).
    ax_b.set_ylim(0, None)
    xlo, xhi = ax_b.get_xlim()
    pad = 0.08 * (xhi - xlo)
    ax_b.set_xlim(xlo - pad, xhi + pad)
    ax_b.set_xlabel("Dose rate (MGy s$^{-1}$)")
    ax_b.set_ylabel("Detection dose /\ncrystal lifetime")

    for ax, letter in zip((ax_a, ax_b), "ab"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_cryo(registry, results, out_path, tnb_window=(395.0, 430.0)):
    """Figure 3: room-temperature film ensemble versus 100 K single crystals.

    The two datasets are not directly commensurable and the figure is built
    to make that visible rather than to hide it.  At room temperature a
    thin film containing many randomly oriented microcrystals is probed
    continuously during irradiation, giving a dose axis.  At 100 K a single
    crystal is measured once before irradiation and again after each of
    several X-ray exposures, giving an exposure index but no calibrated
    dose.  The comparison is therefore between *shapes* and *directions* of
    change, not between rates.

    Parameters
    ----------
    registry
        Experiment registry, for the cryo acquisitions.
    results
        Processed room-temperature conditions.
    out_path
        Output path.
    tnb_window
        TNB integration window, nm.

    Returns
    -------
    matplotlib.figure.Figure
    """

    out_path = Path(out_path)
    wl = results["DTNB_5"].wavelength_nm
    vis = band_mask(wl, 285, 660)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5),
                             gridspec_kw=dict(wspace=0.42))

    # --- (a) cryo difference spectra, per exposure ---
    ax = axes[0]
    shade_band(ax, *tnb_window)
    cryo_bands: dict[str, list[float]] = {}
    for group, files in registry.cryo.items():
        soak = "DTNB" if "DTNB" in group else "apo"
        base = registry.load(files[0]).absorbance[:, 0]
        vals = []
        for j, filename in enumerate(files[1:], start=1):
            after = registry.load(filename).absorbance[:, 0]
            delta = remove_scatter(wl, after - base)
            vals.append(float(integrate_band(wl, delta[:, None], tnb_window)[0]))
            ax.plot(wl[vis], delta[vis], color=SOAK[soak], lw=0.7,
                    alpha=min(0.30 + 0.22 * j, 1.0))
        cryo_bands[group] = vals
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("ΔA")
    ax.annotate("apo (3 crystals)", xy=(0.97, 0.95), xycoords="axes fraction",
                color=SOAK["apo"], fontsize=5.5, va="top", ha="right")
    ax.annotate("DTNB (1 crystal)", xy=(0.97, 0.85), xycoords="axes fraction",
                color=SOAK["DTNB"], fontsize=5.5, va="top", ha="right")

    # --- (b) RT difference spectra at comparable appearance ---
    ax = axes[1]
    shade_band(ax, *tnb_window)
    for key, soak in (("apo_5", "apo"), ("DTNB_5", "DTNB")):
        res = results[key]
        y = res.at_dose(res.max_valid_dose_MGy() * 0.8)
        ax.plot(wl[vis], y[vis], color=SOAK[soak], lw=0.9)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("ΔA")
    ax.annotate("apo", xy=(0.72, 0.62), xycoords="axes fraction",
                color=SOAK["apo"], fontsize=6)
    ax.annotate("DTNB", xy=(0.72, 0.44), xycoords="axes fraction",
                color=SOAK["DTNB"], fontsize=6)

    # --- (c) TNB-window signal: cryo per exposure vs RT ---
    ax = axes[2]
    for group, vals in cryo_bands.items():
        soak = "DTNB" if "DTNB" in group else "apo"
        ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=3, lw=0.9,
                color=SOAK[soak], alpha=0.9)
    rt_marks = []
    for key, soak in (("apo_5", "apo"), ("DTNB_5", "DTNB")):
        res = results[key]
        valid = res.valid()
        series = integrate_band(res.wavelength_nm, res.delta_a[:, :, valid], tnb_window)
        rt_marks.append((soak, float(np.nanmean(series, axis=0)[-1])))
    for soak, value in rt_marks:
        ax.axhline(value, color=SOAK[soak], ls=":", lw=1.0)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("X-ray exposure number")
    ax.set_ylabel("ΔA, 395–430 nm")
    ax.set_xticks([1, 2, 3, 4])
    # Dotted reference lines are colour-matched to the DTNB/apo series they
    # extend (SOAK palette), so no separate on-canvas key is needed; their
    # meaning (room-temperature endpoint, same soak) is stated in the caption.

    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, dx=-0.28)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig
