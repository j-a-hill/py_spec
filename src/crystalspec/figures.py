"""Thesis figures, each built from processed data in a single function.

Every figure takes the processed :class:`~crystalspec.pipeline.ConditionResult`
objects and writes one file.  No figure reads a CSV written by another stage,
so there is exactly one path from the raw ``.asc`` files to any panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.optimize import curve_fit

from .io import read_asc
from .resets import detect_resets, stitch_resets
from .scatter import fit_scatter, remove_scatter
from .style import (
    SOAK,
    SOAK_LABEL,
    TRANSMISSION,
    band_mask,
    check_overlaps,
    panel_label,
    shade_band,
    twin_time_axis,
)

__all__ = ["figure_qc", "figure_signature", "figure_cryo", "figure_dose_dependence", "figure_species", "figure_kinetics", "figure_difference_spectra", "figure_fit_diagnostics", "figure_diagnostic_wavelengths", "figure_experiment_design", "figure_original_idiom", "figure_buffer_controls", "figure_shoulder_isosbestic"]


def figure_qc(registry, results, dose_model, out_path: str | Path):
    """Supplementary quality-control figure.

    Shows the two corrections the analysis depends on -- scattering removal
    and reset repair -- and the resulting usable window for every
    room-temperature film position.

    Parameters
    ----------
    registry
        Experiment registry, for re-reading raw spectra.
    results
        Mapping condition key to :class:`ConditionResult`.
    dose_model
        Unused directly; kept so every figure has one signature.
    out_path
        Where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    demo_key = "DTNB_5"
    wl = results[demo_key].wavelength_nm

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(4, 2, hspace=0.75, wspace=0.32,
                          height_ratios=[1, 1, 1, 1.1])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    axe = fig.add_subplot(gs[2, 0])
    axf = fig.add_subplot(gs[2, 1])
    axg = fig.add_subplot(gs[3, :])

    vis = band_mask(wl, 285, 720)
    acqs = [read_asc(registry.path(f)) for f in registry.conditions[demo_key].files]
    shades = plt.cm.Blues(np.linspace(0.38, 0.88, len(acqs)))

    # (a) raw dark spectra
    for acq, col in zip(acqs, shades):
        axa.plot(wl[vis], acq.absorbance[vis, 0], lw=0.7, color=col)
    axa.set_xlabel("Wavelength (nm)")
    axa.set_ylabel("Absorbance")
    axa.text(0.96, 0.94, "DTNB, 5% T\n5 film positions", transform=axa.transAxes,
             ha="right", va="top", fontsize=6)

    # (b) after scatter removal
    for acq, col in zip(acqs, shades):
        axb.plot(wl[vis], remove_scatter(wl, acq.absorbance[:, 0])[vis], lw=0.7, color=col)
    shade_band(axb, 470, 725)
    axb.axhline(0, color="0.55", lw=0.5)
    axb.set_xlabel("Wavelength (nm)")
    axb.set_ylabel("Absorbance")
    axb.text(0.62, 0.06, "fit window", transform=axb.transAxes, fontsize=6, va="bottom")

    # (c) scatter amplitude per film position
    keys = ["DTNB_5", "DTNB_25", "DTNB_50", "DTNB_100",
            "apo_5", "apo_10", "apo_25", "apo_50", "apo_100"]
    rng = np.random.default_rng(0)
    for i, key in enumerate(keys):
        cond = registry.conditions[key]
        amps = [
            float(fit_scatter(wl, read_asc(registry.path(f)).absorbance[:, 0]).amplitude[0])
            for f in cond.files
        ]
        axc.scatter(np.full(len(amps), i) + rng.uniform(-0.14, 0.14, len(amps)),
                    amps, s=9, lw=0, alpha=0.85, color=SOAK[cond.soak])
        axc.plot([i - 0.26, i + 0.26], [np.median(amps)] * 2, color="0.2", lw=1.2)
    axc.set_xticks(range(len(keys)))
    axc.set_xticklabels([k.replace("_", " ") for k in keys], rotation=90)
    axc.set_ylabel("Scatter amplitude $c$")

    # (d,e) reset detection and stitching on one position
    demo = registry.conditions["DTNB_100"].files[4]
    acq = read_asc(registry.path(demo))
    cycle = acq.cycle_time_s or 0.1
    rep = detect_resets(wl, acq.absorbance, cycle_time_s=cycle)
    stitched = stitch_resets(acq.absorbance, rep)
    t = np.arange(acq.n_spectra) * cycle
    broad = acq.absorbance[band_mask(wl, 500, 650), :].mean(axis=0)

    axd.semilogy(t[1:], np.abs(np.diff(broad)), lw=0.6, color="0.45")
    axd.axhline(rep.threshold, color=SOAK["apo"], ls="--", lw=1)
    if rep.truncated:
        axd.axvline(rep.burst_start_frame * cycle, color="k", lw=1)
    axd.set_ylim(1e-5, 3)
    axd.yaxis.set_minor_formatter(plt.NullFormatter())
    axd.set_xlim(0, 50)
    axd.set_xlabel("Time (s)")
    axd.set_ylabel("|frame-to-frame Δ|")
    axd.text(0.97, 0.90, "– – threshold\n— burst onset", transform=axd.transAxes,
             ha="right", va="top", fontsize=6)

    i412 = int(np.argmin(np.abs(wl - 412)))
    axe.plot(t, acq.absorbance[i412], lw=0.7, color="0.6", label="raw")
    axe.plot(t, stitched[i412], lw=0.9, color=SOAK["DTNB"], label="stitched")
    if rep.truncated:
        axe.axvspan(rep.burst_start_frame * cycle, t[-1], color="0.90", lw=0, zorder=0)
    axe.set_xlim(0, 50)
    axe.set_xlabel("Time (s)")
    axe.set_ylabel("$A_{412}$")
    axe.legend(loc="lower right")

    # (f) per-position difference traces
    res = results[demo_key]
    for i, col in zip(range(res.n_crystals), shades):
        axf.plot(res.dose_MGy, res.delta_a[i, i412, :], lw=0.7, color=col)
    axf.axhline(0, color="0.55", lw=0.5)
    axf.set_xlabel("Dose (MGy)")
    axf.set_ylabel("$\\Delta A_{412}$")

    # (g) usable window per film position
    order = ["apo_100", "DTNB_100", "apo_50", "DTNB_50",
             "apo_25", "DTNB_25", "apo_10", "apo_5", "DTNB_5"]
    for j, key in enumerate(order):
        res = results[key]
        col = SOAK[res.condition.soak]
        for i, tr in enumerate(res.traces):
            y = j + (i - 2) * 0.15
            r = tr.reset_report
            acq_s = read_asc(registry.path(tr.filename)).n_spectra * 0.1
            usable_s = r.usable_frames * 0.1
            axg.plot([0, usable_s], [y, y], color=col, lw=1.4, solid_capstyle="butt")
            if usable_s < acq_s:
                axg.plot([usable_s, acq_s], [y, y], color="0.86", lw=1.4,
                         solid_capstyle="butt")
                axg.plot(usable_s, y, marker="|", color="k", ms=4, mew=0.9)
            if r.stitched_frames.size:
                axg.plot(r.stitched_frames * 0.1,
                         np.full(r.stitched_frames.size, y),
                         ls="none", marker="o", ms=1.6, color="k")
    axg.set_yticks(range(len(order)))
    axg.set_yticklabels([k.replace("_", " ") for k in order])
    axg.set_xlim(0, 100)
    axg.margins(y=0.05)
    axg.set_xlabel("Time (s)")
    axg.text(0.0, 1.015,
             "coloured: analysed    grey: excluded    ● stitched reset    | burst onset",
             transform=axg.transAxes, fontsize=6, va="bottom")

    for ax, letter in zip([axa, axb, axc, axd, axe, axf, axg], "abcdefg"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings

        warnings.warn(f"{out_path.name}: {len(overlaps)} text overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_signature(results, out_path, tnb_window=(395.0, 430.0),
                     dose_fraction=0.8, n_boot=2000, seed=0):
    """Figure 1: the DTNB-specific difference signature.

    Top row: mean difference spectra for DTNB-soaked and apo films at
    matched dose, with the label-specific difference between them.
    Bottom row: the integrated TNB-window signal against dose for both soak
    states, and the difference-of-differences that isolates the label.

    The comparison is always made at *matched absorbed dose*, not matched
    time, because the four attenuator settings deliver dose at rates
    differing twentyfold.

    Parameters
    ----------
    results
        Mapping condition key to :class:`ConditionResult`.
    out_path
        Output path.
    tnb_window
        Wavelength window integrated for the TNB signal, nm.
    dose_fraction
        Fraction of the common usable dose range at which the spectra in the
        top row are shown.
    n_boot, seed
        Bootstrap settings for the confidence bands.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .quantify import band_series, bootstrap_ci, integrate_band

    out_path = Path(out_path)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]
    wl = results[pairs[0][0]].wavelength_nm
    vis = band_mask(wl, 285, 660)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4),
                             sharey="row",
                             gridspec_kw=dict(hspace=0.80, wspace=0.18))

    for col, (dk, ak) in enumerate(pairs):
        d, a = results[dk], results[ak]
        common_max = min(d.max_valid_dose_MGy(), a.max_valid_dose_MGy())
        shown = common_max * dose_fraction

        # --- top: difference spectra at matched dose ---
        ax = axes[0, col]
        yd, ya = d.at_dose(shown), a.at_dose(shown)
        shade_band(ax, *tnb_window)
        ax.plot(wl[vis], ya[vis], color=SOAK["apo"], lw=0.8)
        ax.plot(wl[vis], yd[vis], color=SOAK["DTNB"], lw=0.8)
        ax.plot(wl[vis], (yd - ya)[vis], color="k", lw=1.1)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(f"{d.condition.transmission_pct:.0f}% T\n{shown:.1f} MGy")
        if col == 0:
            ax.set_ylabel("ΔA")
        ax.set_xlabel("Wavelength (nm)")

        # --- bottom: TNB-window signal vs dose ---
        ax = axes[1, col]
        bs_d = band_series(d, tnb_window, "TNB", n_boot=n_boot, seed=seed)
        bs_a = band_series(a, tnb_window, "TNB", n_boot=n_boot, seed=seed)
        for bs, soak in ((bs_a, "apo"), (bs_d, "DTNB")):
            keep = bs.dose_MGy <= common_max
            ax.fill_between(bs.dose_MGy[keep], bs.lo[keep], bs.hi[keep],
                            color=SOAK[soak], alpha=0.22, lw=0)
            ax.plot(bs.dose_MGy[keep], bs.mean[keep], color=SOAK[soak], lw=1.0)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("Dose (MGy)")
        if col == 0:
            ax.set_ylabel("ΔA, 395–430 nm")

    # Direct labels rather than a legend box, in the flat red end of the
    # first panel where no series carries structure.
    ax = axes[0, 0]
    for text, colour, y in (("apo", SOAK["apo"], 0.70),
                            ("DTNB", SOAK["DTNB"], 0.56),
                            ("DTNB − apo", "k", 0.42)):
        ax.annotate(text, xy=(0.52, y), xycoords="axes fraction",
                    color=colour, fontsize=6, ha="left")
    axes[0, 3].annotate("TNB window", xy=(0.30, 0.92), xycoords="axes fraction",
                        fontsize=5.5, color="0.35", ha="left")

    panel_label(axes[0, 0], "a", dx=-0.34)
    panel_label(axes[1, 0], "b", dx=-0.34)

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
    from .quantify import integrate_band

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
    ax.annotate("dotted: room-temperature\nendpoint, same soak",
                xy=(0.04, 0.06), xycoords="axes fraction", fontsize=5.5,
                color="0.35", va="bottom")

    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, dx=-0.28)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_dose_dependence(results, slopes, out_path,
                           tnb_window=(395.0, 430.0), fit_dose_max=2.0,
                           n_boot=2000, seed=0):
    """Figure 2: dose dependence of the TNB signal.

    Panel (a) shows the TNB-window difference signal against absorbed dose,
    DTNB above apo.  Dose is referenced to the shutter opening detected in
    each trace, and each series ends where the sample stopped being a valid
    absorbance sample.  Panel (b) shows the per-position dose-integrated
    signal, the quantity the permutation test is computed on.  Panel (c)
    gives the initial slope of the TNB signal with dose.

    The slope replaces the fitted characteristic dose used earlier.  A
    stretched exponential fitted over the full range drove its shape
    parameter to the imposed bound in every DTNB condition, so the recovered
    dose was not a converged estimate; and the usable range now differs
    tenfold between conditions, which a whole-curve fit handles badly.  A
    slope over a common low-dose window is defined for every condition,
    needs no model, and is directly comparable across dose rates.

    Parameters
    ----------
    results
        Processed conditions.
    slopes
        DataFrame from the initial-slope table, filtered to one band.
    out_path
        Output path.
    tnb_window
        Integration window, nm.
    fit_dose_max
        Upper dose limit of the slope fit, MGy.
    n_boot, seed
        Bootstrap settings.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from itertools import combinations

    from .quantify import band_series, integrate_band

    out_path = Path(out_path)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          hspace=0.66, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- (a) TNB signal vs dose ---
    for dk, ak in pairs:
        d, a = results[dk], results[ak]
        T = int(d.condition.transmission_pct)
        common = min(d.max_valid_dose_MGy(), a.max_valid_dose_MGy())
        for res, style in ((d, "-"), (a, ":")):
            bs = band_series(res, tnb_window, "TNB", n_boot=n_boot, seed=seed)
            keep = (bs.dose_MGy <= common) & (bs.dose_MGy > 0)
            if keep.sum() < 2:
                continue
            ax_a.plot(bs.dose_MGy[keep], bs.mean[keep], style,
                      color=TRANSMISSION[T], lw=1.2 if style == "-" else 0.9)
            if style == "-":
                ax_a.fill_between(bs.dose_MGy[keep], bs.lo[keep], bs.hi[keep],
                                  color=TRANSMISSION[T], alpha=0.16, lw=0)
                ax_a.annotate(f"{T}% T", xy=(bs.dose_MGy[keep][-1], bs.mean[keep][-1]),
                              xytext=(3, 0), textcoords="offset points",
                              color=TRANSMISSION[T], fontsize=6, va="center")
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("Absorbed dose since shutter opening (MGy)")
    ax_a.set_ylabel("\u0394A, 395\u2013430 nm")
    ax_a.set_title("DTNB films lose TNB-window absorbance at every dose rate")
    ax_a.annotate("solid: DTNB      dotted: apo", xy=(0.015, 0.08),
                  xycoords="axes fraction", fontsize=6, color="0.3")

    # --- (b) per-position test statistic ---
    rng = np.random.default_rng(seed)
    for i, (dk, ak) in enumerate(pairs):
        d, a = results[dk], results[ak]
        common = min(d.max_valid_dose_MGy(), a.max_valid_dose_MGy())
        for off, (res, soak) in zip((-0.16, 0.16), ((d, "DTNB"), (a, "apo"))):
            valid = res.valid() & (res.dose_MGy <= common)
            per = integrate_band(res.wavelength_nm, res.delta_a[:, :, valid], tnb_window)
            dose = res.dose_MGy[valid]
            aucs = []
            for row in per:
                good = np.isfinite(row)
                aucs.append(np.trapezoid(row[good], dose[good])
                            if good.sum() > 1 else np.nan)
            aucs = np.asarray(aucs)
            finite = aucs[np.isfinite(aucs)]
            ax_b.scatter(np.full(finite.size, i + off)
                         + rng.uniform(-0.05, 0.05, finite.size),
                         finite, s=11, lw=0, color=SOAK[soak], alpha=0.9)
            if finite.size:
                ax_b.plot([i + off - 0.10, i + off + 0.10], [finite.mean()] * 2,
                          color="0.2", lw=1.1)
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_yscale("symlog", linthresh=0.01)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels([f"{int(results[d].condition.transmission_pct)}%"
                          for d, _ in pairs])
    ax_b.set_xlabel("X-ray transmission")
    ax_b.set_ylabel("\u222b\u0394A d(dose)")
    ax_b.set_title("Per-position test statistic")
    ax_b.annotate("all four separate completely (p = 0.0079)",
                  xy=(0.5, 1.14), xycoords="axes fraction", fontsize=5.5,
                  color="0.35", ha="center")

    # --- (c) initial slope ---
    sl = slopes.set_index("condition")
    for soak, marker, dx in [("DTNB", "o", 0.0), ("apo", "s", 0.04)]:
        keys = [k for k in sl.index if sl.loc[k, "soak"] == soak]
        T = np.array([sl.loc[k, "transmission_pct"] for k in keys], float)
        v = np.array([sl.loc[k, "slope_per_MGy"] for k in keys], float)
        lo = np.array([sl.loc[k, "ci95_lo"] for k in keys], float)
        hi = np.array([sl.loc[k, "ci95_hi"] for k in keys], float)
        order = np.argsort(T)
        ax_c.errorbar(T[order] * (1 + dx), v[order],
                      yerr=[v[order] - lo[order], hi[order] - v[order]],
                      fmt=marker + "-", ms=4, lw=1.0, capsize=2, elinewidth=0.7,
                      color=SOAK[soak], label=SOAK_LABEL.get(soak, soak))
    ax_c.axhline(0, color="0.6", lw=0.5)
    ax_c.set_xscale("log")
    ax_c.set_xticks([5, 10, 25, 50, 100])
    ax_c.set_xticklabels(["5", "10", "25", "50", "100"])
    ax_c.set_xlabel("X-ray transmission (%)")
    ax_c.set_ylabel("Initial slope (per MGy)")
    ax_c.set_title(f"Slope sign inverts with label (\u2264 {fit_dose_max:g} MGy)")
    ax_c.legend(loc="lower left", fontsize=5.5, framealpha=0.9)

    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, letter, dx=-0.10 if ax is ax_a else -0.30)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_species(results, ground_state, loss_spectra, band_fits, band_params, out_path,
                   tnb_window=(395.0, 430.0)):
    """Figure 4: what the absorbing species are, and which one X-rays remove.

    Panel (a) overlays the pre-irradiation spectra of DTNB-soaked and matched
    apo films.  Panel (b) is their difference, the DTNB-dependent ground-state
    absorbance.  Panel (c) is the loss spectrum, DTNB minus apo at the highest
    common dose, with a constrained band fit.  Panel (d) places the fitted
    band centres against the literature positions of the candidate species.

    The panels are ordered so the reader sees the raw overlay before any
    subtraction, since both differences in panels (b) and (c) rest on the
    assumption that matched films carry comparable material in the beam.

    Parameters
    ----------
    results
        Processed conditions.
    ground_state
        Mapping condition key to the smoothed DTNB-minus-apo pre-irradiation
        spectrum on the analysis wavelength grid.
    loss_spectra
        Mapping condition key to the DTNB-minus-apo difference at the highest
        common dose.
    band_fits
        Mapping condition key to the fitted loss-band centre in nm.
    band_params
        Mapping condition key to the full fitted parameter vector
        ``(amplitude, centre, sigma, slope, offset)`` of the loss band, used
        to draw the fit over its own fitting range only.
    out_path
        Output path.
    tnb_window
        Window shaded as the integration region.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    keys = [k for k in ("DTNB_5", "DTNB_25", "DTNB_50", "DTNB_100")
            if k in ground_state]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.52, wspace=0.28)

    wl = results[keys[0]].wavelength_nm
    mask = (wl >= 285) & (wl <= 725)
    x = wl[mask]

    # --- (a) raw pre-irradiation overlay, one representative pair ---
    from .scatter import remove_scatter
    for key, soak in (("DTNB_25", "DTNB"), ("apo_25", "apo")):
        res = results[key]
        spectra = []
        for trace in res.traces:
            spectra.append(trace.raw_dark)
        mean_raw = np.mean(spectra, axis=0)
        corrected = np.asarray(remove_scatter(wl, mean_raw)).ravel()
        ax_a.plot(x, corrected[mask], color=SOAK[soak], lw=1.1,
                  label=SOAK_LABEL.get(soak, soak))
    shade_band(ax_a, tnb_window[0], tnb_window[1])
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("Absorbance")
    ax_a.set_title("Before irradiation, 25% T")
    ax_a.legend(loc="upper right", fontsize=6)

    # --- (b) ground-state difference ---
    for key in keys:
        T = int(results[key].condition.transmission_pct)
        ax_b.plot(x, ground_state[key], color=TRANSMISSION[T], lw=1.0,
                  label=f"{T}%")
    shade_band(ax_b, tnb_window[0], tnb_window[1])
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("\u0394A (DTNB \u2212 apo)")
    ax_b.set_title("DTNB-dependent absorbance")
    ax_b.legend(loc="upper right", fontsize=5.5, ncol=2, title="transmission",
                title_fontsize=5.5)

    # --- (c) loss spectrum with the fitted band overlaid ---
    fit_lo, fit_hi = 360.0, 520.0
    seg = (x >= fit_lo) & (x <= fit_hi)
    for key in keys:
        T = int(results[key].condition.transmission_pct)
        ax_c.plot(x, loss_spectra[key], color=TRANSMISSION[T], lw=1.0)
        params = band_params.get(key)
        if params is not None:
            amp, centre, sigma, slope, offset = params
            xs = x[seg]
            model = -(amp * np.exp(-(xs - centre) ** 2 / (2 * sigma ** 2))
                      + slope * (xs - xs.mean()) / 100 + offset)
            ax_c.plot(xs, model, color="0.25", lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax_c.axvspan(fit_lo, fit_hi, color="0.9", zorder=0)
    ax_c.axhline(0, color="0.6", lw=0.5)
    ax_c.set_xlabel("Wavelength (nm)")
    ax_c.set_ylabel("\u0394A lost")
    ax_c.set_title("What irradiation removes, with band fits")
    ax_c.annotate("dashed: single-band fit\ngrey: fit range",
                  xy=(0.96, 0.08), xycoords="axes fraction", fontsize=5.5,
                  color="0.35", ha="right")

    # --- (d) band positions against literature ---
    refs = [("protein\u2013S\u2013S\u2013TNB", 328.0),
            ("free TNB$^{2-}$", 412.0)]
    for label, pos in refs:
        ax_d.axvline(pos, color="0.55", ls="--", lw=0.8)
        ax_d.annotate(label, xy=(pos, 0.55), xycoords=("data", "axes fraction"),
                      ha="center", fontsize=5.5, color="0.35",
                      rotation=90, va="center",
                      bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    gs_pos = []
    for key in keys:
        seg = (x >= 300) & (x <= 470)
        gs_pos.append(x[seg][int(np.argmax(ground_state[key][seg]))])
    T_vals = [results[k].condition.transmission_pct for k in keys]
    ax_d.scatter(gs_pos, T_vals, s=26, marker="o", color=SOAK["DTNB"],
                 label="ground-state band", zorder=3)
    ax_d.scatter([band_fits[k] for k in keys], T_vals, s=26, marker="D",
                 facecolor="white", edgecolor=SOAK["DTNB"], lw=1.0,
                 label="band lost to X-rays", zorder=3)
    ax_d.set_yscale("log")
    ax_d.set_yticks([5, 25, 50, 100])
    ax_d.set_yticklabels(["5", "25", "50", "100"])
    ax_d.set_xlim(300, 460)
    ax_d.set_xlabel("Band centre (nm)")
    ax_d.set_ylabel("Transmission (%)")
    ax_d.set_title("Observed bands vs literature")
    ax_d.annotate("328 nm masked by the\nprotein UV growth", xy=(0.03, 0.06),
                  xycoords="axes fraction", fontsize=5.5, color="0.35")
    ax_d.legend(loc="lower right", fontsize=5.5, framealpha=0.9)

    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_kinetics(results, dose_model, no_xray, out_path,
                    tnb_window=(395.0, 430.0), uv_window=(300.0, 325.0),
                    min_positions=3, seed=3):
    """Figure 5: kinetics of the label-specific loss, and what drives it.

    Panel (a) plots the DTNB-minus-apo signal in the TNB window against dose
    with biexponential fits and the standard error of the mean as a band.  The
    standard deviation across film positions is drawn as a single bar per
    condition at the right-hand end of each curve, because it is larger than
    the difference it brackets -- absolute amplitudes vary more
    between positions than DTNB differs from apo, since each position carries
    an unknown amount of crystalline material.  The difference is nonetheless
    resolvable because its sign is consistent within each arm, which is what
    the permutation test in table 3 measures.  Panels (b) and (c) replot the same curves referenced
    to their own first frame, against dose and against time since the
    detected onset, to ask which axis the conditions collapse onto: a
    dose-driven process should superimpose in (b), a time-limited one in (c).
    Panel (d) overlays the no-X-ray control.  Its time axis is elapsed time
    from each trace's own zero, which is the detected onset for the irradiated
    films and acquisition start for the control: the control has no onset to
    detect, having never been irradiated.  The two references differ by the
    few seconds of hand-timed delay before the shutter opened, which is small
    against the twenty-five second span shown and does not affect the
    comparison being made -- that one set changes and the other does not.

    The dose rates span twentyfold across conditions, which is what makes the
    dose-versus-time comparison possible at all.  Time is measured from the
    detected onset of change rather than from acquisition start, because the
    shutter was opened by hand at an unlogged interval after recording began.

    Notes
    -----
    The contributing set of film positions is held fixed for the whole trace.
    Averaging over a set that shrinks as positions drop out puts a step in the
    mean at every dropout -- when the position carrying the largest signal
    ends, the mean of the survivors jumps.  At 25 per cent transmission one
    position ends at 10.7 MGy and the naive mean steps by 0.0074 absorbance
    units, twenty-six times the frame-to-frame noise.

    Parameters
    ----------
    results
        Processed conditions.
    dose_model
        Converts transmission to dose rate.
    no_xray
        ``(time_s, delta_a)`` of the scatter-corrected no-X-ray control,
        integrated over ``tnb_window`` and referenced to its own start.  Its
        time base is acquisition start, not a detected onset.
    out_path
        Output path.
    tnb_window, uv_window
        Integration windows, nm.
    min_positions
        Number of film positions the fixed contributing set must retain.
    seed
        Unused placeholder retained for call compatibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .quantify import integrate_band

    out_path = Path(out_path)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    def biexp(D, A1, k1, A2, k2, c):
        return c - A1 * (1 - np.exp(-k1 * D)) - A2 * (1 - np.exp(-k2 * D))

    def _fixed(arr):
        ends = np.array([np.nonzero(np.isfinite(row))[0].max() + 1
                         if np.isfinite(row).any() else 0 for row in arr])
        keep_to = int(np.sort(ends)[::-1][min(min_positions, len(ends)) - 1])
        return np.nonzero(ends >= keep_to)[0], keep_to

    series = {}
    for dkey, akey in pairs:
        dres, ares = results[dkey], results[akey]
        dmax = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
        dv = dres.valid() & (dres.dose_MGy <= dmax)
        av = ares.valid() & (ares.dose_MGy <= dmax)
        pd_ = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, dv], tnb_window)
        pa_ = integrate_band(ares.wavelength_nm, ares.delta_a[:, :, av], tnb_window)
        n = min(pd_.shape[1], pa_.shape[1])
        pd_, pa_ = pd_[:, :n], pa_[:, :n]
        kd, ed = _fixed(pd_)
        ka, ea = _fixed(pa_)
        n = min(ed, ea)
        pd_ = pd_[np.ix_(kd, np.arange(n))]
        pa_ = pa_[np.ix_(ka, np.arange(n))]
        dose = dres.dose_MGy[dv][:n]
        rate = dose_model.rate_at(dres.condition.transmission_pct)
        mean = np.nanmean(pd_, axis=0) - np.nanmean(pa_, axis=0)
        # Standard deviation across film positions.  Not the standard error:
        # the spread between positions is the quantity of interest, because it
        # reflects how much material each position had in the beam.
        sd = np.sqrt(np.nanvar(pd_, axis=0, ddof=1)
                     + np.nanvar(pa_, axis=0, ddof=1))
        se = np.sqrt(np.nanvar(pd_, axis=0, ddof=1) / pd_.shape[0]
                     + np.nanvar(pa_, axis=0, ddof=1) / pa_.shape[0])
        series[dkey] = dict(dose=dose, time=dose / rate, mean=mean, sd=sd,
                            se=se, n_pos=(pd_.shape[0], pa_.shape[0]))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.56, wspace=0.30)

    lo, hi = tnb_window
    ulo, uhi = uv_window

    # --- (a) fits with SD across positions ---
    sd_bars = []
    for key, S in series.items():
        T = int(results[key].condition.transmission_pct)
        good = np.isfinite(S["mean"])
        x, y = S["dose"][good], S["mean"][good]
        se_ = S["se"][good]
        # Thin the mean trace to one point per 0.02 MGy: at 10 Hz the raw trace
        # is ~500 points over this window, which reads as a band rather than a
        # line once four conditions overlap.
        step = max(1, int(len(x) / 120))
        ax_a.fill_between(x[::step], (y - se_)[::step], (y + se_)[::step],
                          color=TRANSMISSION[T], alpha=0.22, lw=0, zorder=2)
        ax_a.plot(x[::step], y[::step], color=TRANSMISSION[T], lw=1.0, zorder=3)
        # The position-to-position spread is an order of magnitude larger than
        # the difference and is shown once, as a bar, rather than as a band
        # that would swamp every curve.
        sd_bars.append((T, float(x[-1]), float(y[-1]), float(S["sd"][good][-1])))
        try:
            popt, _ = curve_fit(biexp, x, y, p0=[0.01, 5.0, 0.01, 0.3, 0.0],
                                bounds=([0, 1e-2, 0, 1e-3, -1],
                                        [1, 200, 1, 10, 1]), maxfev=80000)
            ax_a.plot(x, biexp(x, *popt), color="0.2", lw=0.9,
                      ls=(0, (3, 2)), zorder=4)
        except Exception:
            pass
    ax_a.axhline(0, color="0.6", lw=0.5)
    # Linear axis truncated to the range every condition reaches.  A log axis
    # spanning the full 0.003-23 MGy squeezes the region where the conditions
    # can actually be compared into a fraction of the width; the two long
    # conditions continue beyond this window and are shown in panel (b).
    xmax = min(S["dose"].max() for S in series.values())
    ax_a.set_xlim(0, xmax * 1.14)
    for T, xe, ye, sde in sd_bars:
        ax_a.errorbar([xmax * 1.07], [ye], yerr=[sde], fmt="none",
                      ecolor=TRANSMISSION[T], elinewidth=0.8, capsize=1.5,
                      zorder=4)
    # Scale to the means and the standard error.  The position-to-position
    # standard deviation appears as a single bar at the right of each curve
    # instead of a band: it exceeds the difference itself, so drawn as a band
    # it hides the curves it is meant to qualify.
    lows = [np.nanmin(S["mean"] - S["se"]) for S in series.values()]
    highs = [np.nanmax(S["mean"] + S["se"]) for S in series.values()]
    pad = 0.14 * (max(highs) - min(lows))
    ax_a.set_ylim(min(lows) - pad, max(highs) + pad)
    # In-figure key so panel (a) can be decoded without the caption.  Proxy
    # artists name the encodings; conditions are keyed by colour in panel (b).
    key = [Line2D([], [], color="0.2", lw=0.9, ls=(0, (3, 2)),
                  label="biexponential fit"),
           Patch(facecolor="0.45", alpha=0.22, lw=0, label="s.e.m."),
           Line2D([], [], color="0.45", lw=0.8, marker="_", ls="none",
                  label="s.d. over positions")]
    ax_a.legend(handles=key, loc="lower left", fontsize=5.0, framealpha=0.9,
                handlelength=1.6, borderpad=0.3, labelspacing=0.3)
    ax_a.set_xlabel("Absorbed dose (MGy)")
    ax_a.set_ylabel(f"\u0394A, DTNB \u2212 apo\n({lo:.0f}\u2013{hi:.0f} nm)")


    # --- (b) dose axis, (c) time-since-onset axis ---
    for ax, xkey, xlabel in (
        (ax_b, "dose", "Absorbed dose (MGy)"),
        (ax_c, "time", "Time since onset of change (s)"),
    ):
        for key, S in series.items():
            T = int(results[key].condition.transmission_pct)
            y = S["mean"] - S["mean"][0]
            good = np.isfinite(y)
            ax.plot(S[xkey][good], y[good], color=TRANSMISSION[T], lw=1.1,
                    label=f"{T}%")
        ax.axhline(0, color="0.6", lw=0.5)
        if xkey == "time":
            ax.set_xlim(0, 4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"\u0394A change ({lo:.0f}\u2013{hi:.0f} nm)")
    ax_b.legend(loc="upper right", fontsize=5.5, ncol=2, title="transmission",
                title_fontsize=5.5, framealpha=0.9)

    # --- (d) no-X-ray control ---
    t_ctrl, y_ctrl = no_xray
    for key, S in series.items():
        T = int(results[key].condition.transmission_pct)
        y = S["mean"] - S["mean"][0]
        good = np.isfinite(y)
        ax_d.plot(S["time"][good], y[good], color=TRANSMISSION[T], lw=1.0)
    ax_d.plot(t_ctrl, y_ctrl - y_ctrl[0], color="0.25", lw=1.4,
              label="no X-rays")
    ax_d.axhline(0, color="0.6", lw=0.5)
    ax_d.set_xlim(0, 25)
    ax_d.set_xlabel("Elapsed time (s)")
    ax_d.set_ylabel(f"\u0394A change ({lo:.0f}\u2013{hi:.0f} nm)")
    ax_d.legend(loc="lower right", fontsize=5.5)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_difference_spectra(results, out_path, tnb_window=(395.0, 430.0),
                              uv_window=(300.0, 325.0),
                              zero_window=(470.0, 500.0), seed=5, n_boot=3000):
    """Figure 6: the full difference spectra behind the band measurements.

    Every other figure reduces the spectra to a number integrated over one
    window.  This one shows the spectra themselves, so the reader can see that
    the negative TNB-window feature and the large positive ultraviolet feature
    are separate, and judge the assignments directly.

    Panel (a) overlays the DTNB and apo difference spectra at the highest
    common dose for one condition, with the analysis windows shaded.  Panel
    (b) shows their difference for all four transmissions.  Panel (c) resolves
    the difference into bands with the standard deviation across film
    positions, distinguishing features that clear the baseline floor from
    those that do not.  Panel (d) traces the positive ultraviolet feature and
    the negative TNB feature against dose on the same axes.

    The positive ultraviolet feature is far larger than the negative one and
    grows in both arms, so it is not label-specific; it is shown because it
    dominates the spectrum and because it is what masks the mixed-disulphide
    position at 328 nm.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    tnb_window, uv_window, zero_window
        Analysis windows, nm.  ``zero_window`` is the internal-zero region
        used to set the baseline floor.
    seed, n_boot
        Bootstrap settings for panel (c).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import savgol_filter
    from .quantify import integrate_band

    out_path = Path(out_path)
    rng = np.random.default_rng(seed)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    wl = results["DTNB_5"].wavelength_nm
    keep = (wl >= 288.0) & (wl <= 720.0)
    x = wl[keep]

    def endpoint_pair(dkey, akey):
        dres, ares = results[dkey], results[akey]
        dmax = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
        i = np.nonzero(dres.valid() & (dres.dose_MGy <= dmax))[0][-1]
        j = np.nonzero(ares.valid() & (ares.dose_MGy <= dmax))[0][-1]
        sd = savgol_filter(np.nanmean(dres.delta_a[:, :, i], axis=0)[keep], 61, 3)
        sa = savgol_filter(np.nanmean(ares.delta_a[:, :, j], axis=0)[keep], 61, 3)
        return sd, sa, dmax

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.62, wspace=0.32)

    # --- (a) the two arms, one condition ---
    dkey, akey = "DTNB_25", "apo_25"
    sd, sa, dmax = endpoint_pair(dkey, akey)
    for w in (uv_window, tnb_window):
        shade_band(ax_a, w[0], w[1])
    ax_a.plot(x, sa, color=SOAK["apo"], lw=1.2, label="apo")
    ax_a.plot(x, sd, color=SOAK["DTNB"], lw=1.2, label="DTNB")
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("\u0394A from pre-irradiation")
    ax_a.legend(loc="upper right", fontsize=5.5)

    # --- (b) the difference, all four ---
    for dk, ak in pairs:
        T = int(results[dk].condition.transmission_pct)
        d_, a_, dm_ = endpoint_pair(dk, ak)
        ax_b.plot(x, d_ - a_, color=TRANSMISSION[T], lw=1.1,
                  label=f"{T}% ({dm_:.1f} MGy)")
    for w in (uv_window, tnb_window):
        shade_band(ax_b, w[0], w[1])
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("\u0394A, DTNB \u2212 apo")
    ax_b.legend(loc="lower right", fontsize=4.8, ncol=1)

    # --- (c) band-resolved, with baseline floor ---
    bands = [("290\u2013300", (290.0, 300.0)), (f"{uv_window[0]:.0f}\u2013{uv_window[1]:.0f}", uv_window),
             ("325\u2013360", (325.0, 360.0)), (f"{tnb_window[0]:.0f}\u2013{tnb_window[1]:.0f}", tnb_window),
             ("520\u2013560", (520.0, 560.0)), (f"{zero_window[0]:.0f}\u2013{zero_window[1]:.0f}", zero_window)]
    width = 0.20
    floor = []
    for bi, (bname, (blo, bhi)) in enumerate(bands):
        for ci, (dk, ak) in enumerate(pairs):
            T = int(results[dk].condition.transmission_pct)
            dres, ares = results[dk], results[ak]
            dmx = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
            i = np.nonzero(dres.valid() & (dres.dose_MGy <= dmx))[0][-1]
            j = np.nonzero(ares.valid() & (ares.dose_MGy <= dmx))[0][-1]
            P = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, [i]], (blo, bhi))[:, 0]
            Q = integrate_band(ares.wavelength_nm, ares.delta_a[:, :, [j]], (blo, bhi))[:, 0]
            P, Q = P[np.isfinite(P)], Q[np.isfinite(Q)]
            if P.size < 2 or Q.size < 2:
                continue
            val = P.mean() - Q.mean()
            err = np.sqrt(P.var(ddof=1) / P.size + Q.var(ddof=1) / Q.size)
            ax_c.bar(bi + (ci - 1.5) * width, val, width * 0.9,
                     yerr=err, color=TRANSMISSION[T], lw=0,
                     error_kw=dict(lw=0.5, capsize=1.0, ecolor="0.35"))
            if (blo, bhi) == zero_window:
                floor.append(abs(val) + err)
    if floor:
        f = max(floor)
        ax_c.axhspan(-f, f, color="0.5", alpha=0.16, lw=0, zorder=0)
    ax_c.axhline(0, color="0.4", lw=0.6)
    ax_c.set_xticks(range(len(bands)))
    ax_c.set_xticklabels([b[0] for b in bands], fontsize=5.0, rotation=30, ha="right")
    ax_c.set_xlabel("Band (nm)")
    ax_c.set_ylabel("\u0394A, DTNB \u2212 apo")
    ax_c.set_ylim(top=0.105)

    # --- (d) positive and negative features vs dose ---
    for dk, ak in pairs[:2]:
        T = int(results[dk].condition.transmission_pct)
        dres, ares = results[dk], results[ak]
        dmx = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
        dv = dres.valid() & (dres.dose_MGy <= dmx)
        dose = dres.dose_MGy[dv]
        uv = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, dv], uv_window)
        tn = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, dv], tnb_window)
        # Hold the contributing positions fixed, as in figure_kinetics: a
        # shrinking set steps the mean at every dropout.
        ends = np.array([np.nonzero(np.isfinite(r))[0].max() + 1
                         if np.isfinite(r).any() else 0 for r in tn])
        cut = int(np.sort(ends)[::-1][min(3, len(ends)) - 1])
        sel = np.nonzero(ends >= cut)[0]
        uv, tn, dose = uv[sel, :cut], tn[sel, :cut], dose[:cut]
        ax_d.plot(dose, np.nanmean(uv, axis=0), color=TRANSMISSION[T], lw=1.2,
                  label=f"{uv_window[0]:.0f}\u2013{uv_window[1]:.0f} nm, {T}%")
        ax_d.plot(dose, np.nanmean(tn, axis=0), color=TRANSMISSION[T], lw=1.2,
                  ls=(0, (2, 1.5)),
                  label=f"{tnb_window[0]:.0f}\u2013{tnb_window[1]:.0f} nm, {T}%")
    ax_d.axhline(0, color="0.6", lw=0.5)
    ax_d.set_xlabel("Absorbed dose (MGy)")
    ax_d.set_ylabel("\u0394A (DTNB films)")
    ax_d.legend(loc="center right", fontsize=4.6, ncol=1, framealpha=0.9)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_fit_diagnostics(results, out_path, fit_window=(292.0, 470.0),
                           growth_window=(292.0, 340.0),
                           tnb_window=(395.0, 430.0), savgol=(61, 3),
                           calibration=None):
    """Figure S2: every processing step and fit behind the band assignments.

    Built so the fits can be audited rather than taken on trust.  Panel (a)
    overlays raw and smoothed spectra with the residual of smoothing, so the
    effect of the filter is visible.  Panel (b) shows the two-Gaussian
    decomposition with its components and residual.  Panel (c) is the reason
    the fitted band centre cannot be trusted: wavelength-to-wavelength noise
    against wavelength, which rises steeply into the blue where the growth
    band's flank lies.  Panel (d) shows how the fitted DTNB-minus-apo centre
    shift depends on where the fit window starts, next to two model-free
    measures of the same quantity.

    Panels (c) and (d) exist because the fitted shift is not robust: it falls
    from about eight nanometres to under three as the fit window is moved off
    the noisy blue flank, and the model-free measures do not reproduce it.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    fit_window, growth_window, tnb_window
        Fit range, growth-band search range, and TNB integration window, nm.
    savgol
        ``(window_length, polyorder)`` of the Savitzky-Golay filter applied
        before peak finding.  Fits are performed on filtered spectra; panel (a)
        quantifies what that costs.
    calibration
        Optional ``(angles_deg, lambda_max_nm)`` from an external
        dihedral-to-absorption calibration, annotated on panel (d).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import savgol_filter
    from scipy.optimize import curve_fit

    out_path = Path(out_path)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"), ("DTNB_50", "apo_50")]
    wl = results["DTNB_5"].wavelength_nm
    keep = (wl >= 285.0) & (wl <= 725.0)
    x = wl[keep]

    def gauss(v, A, c, f):
        return A * np.exp(-4 * np.log(2) * ((v - c) / f) ** 2)

    def two_gauss(v, A1, c1, f1, A2, c2, f2):
        return gauss(v, A1, c1, f1) + gauss(v, A2, c2, f2)

    def endpoint(key):
        r = results[key]
        i = int(np.nonzero(r.valid() & (r.dose_MGy <= r.max_valid_dose_MGy()))[0][-1])
        return r, i

    def mean_spectrum(key, smooth=True):
        r, i = endpoint(key)
        raw = np.nanmean(r.delta_a[:, :, i], axis=0)[keep]
        return savgol_filter(raw, *savgol) if smooth else raw

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.60, wspace=0.32)

    # --- (a) smoothing: raw, filtered, residual ---
    raw = mean_spectrum("DTNB_25", smooth=False)
    sm = mean_spectrum("DTNB_25", smooth=True)
    ax_a.plot(x, raw, color="0.65", lw=0.5, label="raw")
    ax_a.plot(x, sm, color=SOAK["DTNB"], lw=1.1, label="filtered")
    ax_a.plot(x, raw - sm, color="0.25", lw=0.5, label="residual")
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("\u0394A")
    ax_a.legend(loc="upper right", fontsize=5.2, framealpha=0.9)

    # --- (b) the fit and its components ---
    flo, fhi = fit_window
    fw = (x >= flo) & (x <= fhi)
    xf, yf = x[fw], sm[fw]
    p0 = [yf.max(), 310.0, 50.0, yf.max() * 0.3, 360.0, 80.0]
    bounds = ([0, 295, 20, 0, 330, 30], [1, 330, 90, 1, 430, 200])
    popt, _ = curve_fit(two_gauss, xf, yf, p0=p0, bounds=bounds, maxfev=200000)
    ax_b.plot(xf, yf, color=SOAK["DTNB"], lw=1.1, label="data")
    ax_b.plot(xf, two_gauss(xf, *popt), color="0.2", lw=0.9, ls=(0, (3, 2)),
              label="two-Gaussian fit")
    ax_b.plot(xf, gauss(xf, *popt[:3]), color="0.45", lw=0.7, ls=(0, (1, 1.5)),
              label=f"band 1, {popt[1]:.0f} nm")
    ax_b.plot(xf, gauss(xf, *popt[3:]), color="0.65", lw=0.7, ls=(0, (1, 1.5)),
              label=f"band 2, {popt[4]:.0f} nm")
    ax_b.plot(xf, yf - two_gauss(xf, *popt), color="0.25", lw=0.5,
              label="residual")
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("\u0394A")
    ax_b.legend(loc="upper right", fontsize=4.6, framealpha=0.9)

    # --- (c) noise against wavelength ---
    edges = np.arange(285.0, 726.0, 15.0)
    for dk, ak in pairs[:2]:
        for key, colour in ((dk, SOAK["DTNB"]), (ak, SOAK["apo"])):
            r, i = endpoint(key)
            centres, noise = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                bw = (wl >= lo) & (wl <= hi)
                vals = [np.std(np.diff(r.delta_a[pi, bw, i])) / np.sqrt(2)
                        for pi in range(r.delta_a.shape[0])
                        if np.isfinite(r.delta_a[pi, bw, i]).all()]
                if vals:
                    centres.append(0.5 * (lo + hi))
                    noise.append(np.mean(vals))
            ax_c.plot(centres, noise, color=colour, lw=1.0,
                      alpha=0.55 if dk.endswith("_5") else 1.0)
    shade_band(ax_c, *growth_window)
    shade_band(ax_c, *tnb_window)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Wavelength (nm)")
    ax_c.set_ylabel("Noise (\u0394A per point)")
    key_c = [Line2D([], [], color=SOAK["DTNB"], lw=1.0, label="DTNB"),
             Line2D([], [], color=SOAK["apo"], lw=1.0, label="apo")]
    ax_c.legend(handles=key_c, loc="upper right", fontsize=5.2, framealpha=0.9)

    # --- (d) window sensitivity of the fitted shift ---
    starts = np.arange(292.0, 306.1, 2.0)
    for dk, ak in pairs:
        T = int(results[dk].condition.transmission_pct)
        shifts = []
        for wlo in starts:
            fwv = (x >= wlo) & (x <= fhi)
            cs = []
            for key in (dk, ak):
                yy = mean_spectrum(key)[fwv]
                bb = ([0, max(295, wlo), 20, 0, 330, 30],
                      [1, 330, 90, 1, 430, 200])
                try:
                    pp, _ = curve_fit(two_gauss, x[fwv], yy,
                                      p0=[yy.max(), 310, 50, yy.max() * 0.3, 360, 80],
                                      bounds=bb, maxfev=200000)
                    cs.append(pp[1])
                except Exception:
                    cs.append(np.nan)
            shifts.append(cs[0] - cs[1])
        ax_d.plot(starts, shifts, color=TRANSMISSION[T], lw=1.1,
                  marker="o", ms=2.2, label=f"{T}%")
    ax_d.axhline(0, color="0.4", lw=0.6)
    ax_d.set_xlabel("Fit window start (nm)")
    ax_d.set_ylabel("Fitted DTNB \u2212 apo\ncentre shift (nm)")
    ax_d.legend(loc="lower right", fontsize=5.2, ncol=3, framealpha=0.9,
                title="transmission", title_fontsize=5.2)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


DIAGNOSTIC_NM = {310.0: "growth band",
                 328.0: "protein\u2013S\u2013S\u2013TNB",
                 400.0: "disulphide radical",
                 412.0: "free TNB$^{2-}$",
                 480.0: "isosbestic",
                 580.0: "solvated e$^-$"}


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
    from .quantify import diagnostic_trace, fit_single_exponential

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
    ax_a.legend(loc="upper right", fontsize=5.5, framealpha=0.9)

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
    ax_b.legend(loc="lower left", fontsize=5.0, ncol=2, framealpha=0.85,
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
            lab = (f"{T}%, $D_{{90}}$ {fit['D90_MGy']:.1f} MGy"
                   f" ($R^2$ {fit['R2_single']:.2f})")
        elif np.isfinite(fit["D90_MGy"]):
            lab = f"{T}%, $D_{{90}}$ > {fit['D90_MGy']:.1f} MGy (not saturated)"
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


def figure_kinetics(results, dose_model, no_xray, out_path,
                    tnb_window=(395.0, 430.0), uv_window=(300.0, 325.0),
                    min_positions=3, seed=3):
    """Figure 5: kinetics of the label-specific loss, and what drives it.

    Panel (a) plots the DTNB-minus-apo signal in the TNB window against dose
    with biexponential fits and the standard error of the mean as a band.  The
    standard deviation across film positions is drawn as a single bar per
    condition at the right-hand end of each curve, because it is larger than
    the difference it brackets -- absolute amplitudes vary more
    between positions than DTNB differs from apo, since each position carries
    an unknown amount of crystalline material.  The difference is nonetheless
    resolvable because its sign is consistent within each arm, which is what
    the permutation test in table 3 measures.  Panels (b) and (c) replot the same curves referenced
    to their own first frame, against dose and against time since the
    detected onset, to ask which axis the conditions collapse onto: a
    dose-driven process should superimpose in (b), a time-limited one in (c).
    Panel (d) overlays the no-X-ray control.  Its time axis is elapsed time
    from each trace's own zero, which is the detected onset for the irradiated
    films and acquisition start for the control: the control has no onset to
    detect, having never been irradiated.  The two references differ by the
    few seconds of hand-timed delay before the shutter opened, which is small
    against the twenty-five second span shown and does not affect the
    comparison being made -- that one set changes and the other does not.

    The dose rates span twentyfold across conditions, which is what makes the
    dose-versus-time comparison possible at all.  Time is measured from the
    detected onset of change rather than from acquisition start, because the
    shutter was opened by hand at an unlogged interval after recording began.

    Notes
    -----
    The contributing set of film positions is held fixed for the whole trace.
    Averaging over a set that shrinks as positions drop out puts a step in the
    mean at every dropout -- when the position carrying the largest signal
    ends, the mean of the survivors jumps.  At 25 per cent transmission one
    position ends at 10.7 MGy and the naive mean steps by 0.0074 absorbance
    units, twenty-six times the frame-to-frame noise.

    Parameters
    ----------
    results
        Processed conditions.
    dose_model
        Converts transmission to dose rate.
    no_xray
        ``(time_s, delta_a)`` of the scatter-corrected no-X-ray control,
        integrated over ``tnb_window`` and referenced to its own start.  Its
        time base is acquisition start, not a detected onset.
    out_path
        Output path.
    tnb_window, uv_window
        Integration windows, nm.
    min_positions
        Number of film positions the fixed contributing set must retain.
    seed
        Unused placeholder retained for call compatibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .quantify import integrate_band

    out_path = Path(out_path)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    def biexp(D, A1, k1, A2, k2, c):
        return c - A1 * (1 - np.exp(-k1 * D)) - A2 * (1 - np.exp(-k2 * D))

    def _fixed(arr):
        ends = np.array([np.nonzero(np.isfinite(row))[0].max() + 1
                         if np.isfinite(row).any() else 0 for row in arr])
        keep_to = int(np.sort(ends)[::-1][min(min_positions, len(ends)) - 1])
        return np.nonzero(ends >= keep_to)[0], keep_to

    series = {}
    for dkey, akey in pairs:
        dres, ares = results[dkey], results[akey]
        dmax = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
        dv = dres.valid() & (dres.dose_MGy <= dmax)
        av = ares.valid() & (ares.dose_MGy <= dmax)
        pd_ = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, dv], tnb_window)
        pa_ = integrate_band(ares.wavelength_nm, ares.delta_a[:, :, av], tnb_window)
        n = min(pd_.shape[1], pa_.shape[1])
        pd_, pa_ = pd_[:, :n], pa_[:, :n]
        kd, ed = _fixed(pd_)
        ka, ea = _fixed(pa_)
        n = min(ed, ea)
        pd_ = pd_[np.ix_(kd, np.arange(n))]
        pa_ = pa_[np.ix_(ka, np.arange(n))]
        dose = dres.dose_MGy[dv][:n]
        rate = dose_model.rate_at(dres.condition.transmission_pct)
        mean = np.nanmean(pd_, axis=0) - np.nanmean(pa_, axis=0)
        # Standard deviation across film positions.  Not the standard error:
        # the spread between positions is the quantity of interest, because it
        # reflects how much material each position had in the beam.
        sd = np.sqrt(np.nanvar(pd_, axis=0, ddof=1)
                     + np.nanvar(pa_, axis=0, ddof=1))
        se = np.sqrt(np.nanvar(pd_, axis=0, ddof=1) / pd_.shape[0]
                     + np.nanvar(pa_, axis=0, ddof=1) / pa_.shape[0])
        series[dkey] = dict(dose=dose, time=dose / rate, mean=mean, sd=sd,
                            se=se, n_pos=(pd_.shape[0], pa_.shape[0]))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.56, wspace=0.30)

    lo, hi = tnb_window
    ulo, uhi = uv_window

    # --- (a) fits with SD across positions ---
    sd_bars = []
    for key, S in series.items():
        T = int(results[key].condition.transmission_pct)
        good = np.isfinite(S["mean"])
        x, y = S["dose"][good], S["mean"][good]
        se_ = S["se"][good]
        # Thin the mean trace to one point per 0.02 MGy: at 10 Hz the raw trace
        # is ~500 points over this window, which reads as a band rather than a
        # line once four conditions overlap.
        step = max(1, int(len(x) / 120))
        ax_a.fill_between(x[::step], (y - se_)[::step], (y + se_)[::step],
                          color=TRANSMISSION[T], alpha=0.22, lw=0, zorder=2)
        ax_a.plot(x[::step], y[::step], color=TRANSMISSION[T], lw=1.0, zorder=3)
        # The position-to-position spread is an order of magnitude larger than
        # the difference and is shown once, as a bar, rather than as a band
        # that would swamp every curve.
        sd_bars.append((T, float(x[-1]), float(y[-1]), float(S["sd"][good][-1])))
        try:
            popt, _ = curve_fit(biexp, x, y, p0=[0.01, 5.0, 0.01, 0.3, 0.0],
                                bounds=([0, 1e-2, 0, 1e-3, -1],
                                        [1, 200, 1, 10, 1]), maxfev=80000)
            ax_a.plot(x, biexp(x, *popt), color="0.2", lw=0.9,
                      ls=(0, (3, 2)), zorder=4)
        except Exception:
            pass
    ax_a.axhline(0, color="0.6", lw=0.5)
    # Linear axis truncated to the range every condition reaches.  A log axis
    # spanning the full 0.003-23 MGy squeezes the region where the conditions
    # can actually be compared into a fraction of the width; the two long
    # conditions continue beyond this window and are shown in panel (b).
    xmax = min(S["dose"].max() for S in series.values())
    ax_a.set_xlim(0, xmax * 1.14)
    for T, xe, ye, sde in sd_bars:
        ax_a.errorbar([xmax * 1.07], [ye], yerr=[sde], fmt="none",
                      ecolor=TRANSMISSION[T], elinewidth=0.8, capsize=1.5,
                      zorder=4)
    # Scale to the means and the standard error.  The position-to-position
    # standard deviation appears as a single bar at the right of each curve
    # instead of a band: it exceeds the difference itself, so drawn as a band
    # it hides the curves it is meant to qualify.
    lows = [np.nanmin(S["mean"] - S["se"]) for S in series.values()]
    highs = [np.nanmax(S["mean"] + S["se"]) for S in series.values()]
    pad = 0.14 * (max(highs) - min(lows))
    ax_a.set_ylim(min(lows) - pad, max(highs) + pad)
    # In-figure key so panel (a) can be decoded without the caption.  Proxy
    # artists name the encodings; conditions are keyed by colour in panel (b).
    key = [Line2D([], [], color="0.2", lw=0.9, ls=(0, (3, 2)),
                  label="biexponential fit"),
           Patch(facecolor="0.45", alpha=0.22, lw=0, label="s.e.m."),
           Line2D([], [], color="0.45", lw=0.8, marker="_", ls="none",
                  label="s.d. over positions")]
    ax_a.legend(handles=key, loc="lower left", fontsize=5.0, framealpha=0.9,
                handlelength=1.6, borderpad=0.3, labelspacing=0.3)
    ax_a.set_xlabel("Absorbed dose (MGy)")
    ax_a.set_ylabel(f"\u0394A, DTNB \u2212 apo\n({lo:.0f}\u2013{hi:.0f} nm)")


    # --- (b) dose axis, (c) time-since-onset axis ---
    for ax, xkey, xlabel in (
        (ax_b, "dose", "Absorbed dose (MGy)"),
        (ax_c, "time", "Time since onset of change (s)"),
    ):
        for key, S in series.items():
            T = int(results[key].condition.transmission_pct)
            y = S["mean"] - S["mean"][0]
            good = np.isfinite(y)
            ax.plot(S[xkey][good], y[good], color=TRANSMISSION[T], lw=1.1,
                    label=f"{T}%")
        ax.axhline(0, color="0.6", lw=0.5)
        if xkey == "time":
            ax.set_xlim(0, 4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"\u0394A change ({lo:.0f}\u2013{hi:.0f} nm)")
    ax_b.legend(loc="upper right", fontsize=5.5, ncol=2, title="transmission",
                title_fontsize=5.5, framealpha=0.9)

    # --- (d) no-X-ray control ---
    t_ctrl, y_ctrl = no_xray
    for key, S in series.items():
        T = int(results[key].condition.transmission_pct)
        y = S["mean"] - S["mean"][0]
        good = np.isfinite(y)
        ax_d.plot(S["time"][good], y[good], color=TRANSMISSION[T], lw=1.0)
    ax_d.plot(t_ctrl, y_ctrl - y_ctrl[0], color="0.25", lw=1.4,
              label="no X-rays")
    ax_d.axhline(0, color="0.6", lw=0.5)
    ax_d.set_xlim(0, 25)
    ax_d.set_xlabel("Elapsed time (s)")
    ax_d.set_ylabel(f"\u0394A change ({lo:.0f}\u2013{hi:.0f} nm)")
    ax_d.legend(loc="lower right", fontsize=5.5)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_difference_spectra(results, out_path, tnb_window=(395.0, 430.0),
                              uv_window=(300.0, 325.0),
                              zero_window=(470.0, 500.0), seed=5, n_boot=3000):
    """Figure 6: the full difference spectra behind the band measurements.

    Every other figure reduces the spectra to a number integrated over one
    window.  This one shows the spectra themselves, so the reader can see that
    the negative TNB-window feature and the large positive ultraviolet feature
    are separate, and judge the assignments directly.

    Panel (a) overlays the DTNB and apo difference spectra at the highest
    common dose for one condition, with the analysis windows shaded.  Panel
    (b) shows their difference for all four transmissions.  Panel (c) resolves
    the difference into bands with the standard deviation across film
    positions, distinguishing features that clear the baseline floor from
    those that do not.  Panel (d) traces the positive ultraviolet feature and
    the negative TNB feature against dose on the same axes.

    The positive ultraviolet feature is far larger than the negative one and
    grows in both arms, so it is not label-specific; it is shown because it
    dominates the spectrum and because it is what masks the mixed-disulphide
    position at 328 nm.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    tnb_window, uv_window, zero_window
        Analysis windows, nm.  ``zero_window`` is the internal-zero region
        used to set the baseline floor.
    seed, n_boot
        Bootstrap settings for panel (c).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import savgol_filter
    from .quantify import integrate_band

    out_path = Path(out_path)
    rng = np.random.default_rng(seed)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
             ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]

    wl = results["DTNB_5"].wavelength_nm
    keep = (wl >= 288.0) & (wl <= 720.0)
    x = wl[keep]

    def endpoint_pair(dkey, akey):
        dres, ares = results[dkey], results[akey]
        dmax = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
        i = np.nonzero(dres.valid() & (dres.dose_MGy <= dmax))[0][-1]
        j = np.nonzero(ares.valid() & (ares.dose_MGy <= dmax))[0][-1]
        sd = savgol_filter(np.nanmean(dres.delta_a[:, :, i], axis=0)[keep], 61, 3)
        sa = savgol_filter(np.nanmean(ares.delta_a[:, :, j], axis=0)[keep], 61, 3)
        return sd, sa, dmax

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.62, wspace=0.32)

    # --- (a) the two arms, one condition ---
    dkey, akey = "DTNB_25", "apo_25"
    sd, sa, dmax = endpoint_pair(dkey, akey)
    for w in (uv_window, tnb_window):
        shade_band(ax_a, w[0], w[1])
    ax_a.plot(x, sa, color=SOAK["apo"], lw=1.2, label="apo")
    ax_a.plot(x, sd, color=SOAK["DTNB"], lw=1.2, label="DTNB")
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("\u0394A from pre-irradiation")
    ax_a.legend(loc="upper right", fontsize=5.5)

    # --- (b) the difference, all four ---
    for dk, ak in pairs:
        T = int(results[dk].condition.transmission_pct)
        d_, a_, dm_ = endpoint_pair(dk, ak)
        ax_b.plot(x, d_ - a_, color=TRANSMISSION[T], lw=1.1,
                  label=f"{T}% ({dm_:.1f} MGy)")
    for w in (uv_window, tnb_window):
        shade_band(ax_b, w[0], w[1])
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("\u0394A, DTNB \u2212 apo")
    ax_b.legend(loc="lower right", fontsize=4.8, ncol=1)

    # --- (c) band-resolved, with baseline floor ---
    bands = [("290\u2013300", (290.0, 300.0)), (f"{uv_window[0]:.0f}\u2013{uv_window[1]:.0f}", uv_window),
             ("325\u2013360", (325.0, 360.0)), (f"{tnb_window[0]:.0f}\u2013{tnb_window[1]:.0f}", tnb_window),
             ("520\u2013560", (520.0, 560.0)), (f"{zero_window[0]:.0f}\u2013{zero_window[1]:.0f}", zero_window)]
    width = 0.20
    floor = []
    for bi, (bname, (blo, bhi)) in enumerate(bands):
        for ci, (dk, ak) in enumerate(pairs):
            T = int(results[dk].condition.transmission_pct)
            dres, ares = results[dk], results[ak]
            dmx = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
            i = np.nonzero(dres.valid() & (dres.dose_MGy <= dmx))[0][-1]
            j = np.nonzero(ares.valid() & (ares.dose_MGy <= dmx))[0][-1]
            P = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, [i]], (blo, bhi))[:, 0]
            Q = integrate_band(ares.wavelength_nm, ares.delta_a[:, :, [j]], (blo, bhi))[:, 0]
            P, Q = P[np.isfinite(P)], Q[np.isfinite(Q)]
            if P.size < 2 or Q.size < 2:
                continue
            val = P.mean() - Q.mean()
            err = np.sqrt(P.var(ddof=1) / P.size + Q.var(ddof=1) / Q.size)
            ax_c.bar(bi + (ci - 1.5) * width, val, width * 0.9,
                     yerr=err, color=TRANSMISSION[T], lw=0,
                     error_kw=dict(lw=0.5, capsize=1.0, ecolor="0.35"))
            if (blo, bhi) == zero_window:
                floor.append(abs(val) + err)
    if floor:
        f = max(floor)
        ax_c.axhspan(-f, f, color="0.5", alpha=0.16, lw=0, zorder=0)
    ax_c.axhline(0, color="0.4", lw=0.6)
    ax_c.set_xticks(range(len(bands)))
    ax_c.set_xticklabels([b[0] for b in bands], fontsize=5.0, rotation=30, ha="right")
    ax_c.set_xlabel("Band (nm)")
    ax_c.set_ylabel("\u0394A, DTNB \u2212 apo")
    ax_c.set_ylim(top=0.105)

    # --- (d) positive and negative features vs dose ---
    for dk, ak in pairs[:2]:
        T = int(results[dk].condition.transmission_pct)
        dres, ares = results[dk], results[ak]
        dmx = min(dres.max_valid_dose_MGy(), ares.max_valid_dose_MGy())
        dv = dres.valid() & (dres.dose_MGy <= dmx)
        dose = dres.dose_MGy[dv]
        uv = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, dv], uv_window)
        tn = integrate_band(dres.wavelength_nm, dres.delta_a[:, :, dv], tnb_window)
        # Hold the contributing positions fixed, as in figure_kinetics: a
        # shrinking set steps the mean at every dropout.
        ends = np.array([np.nonzero(np.isfinite(r))[0].max() + 1
                         if np.isfinite(r).any() else 0 for r in tn])
        cut = int(np.sort(ends)[::-1][min(3, len(ends)) - 1])
        sel = np.nonzero(ends >= cut)[0]
        uv, tn, dose = uv[sel, :cut], tn[sel, :cut], dose[:cut]
        ax_d.plot(dose, np.nanmean(uv, axis=0), color=TRANSMISSION[T], lw=1.2,
                  label=f"{uv_window[0]:.0f}\u2013{uv_window[1]:.0f} nm, {T}%")
        ax_d.plot(dose, np.nanmean(tn, axis=0), color=TRANSMISSION[T], lw=1.2,
                  ls=(0, (2, 1.5)),
                  label=f"{tnb_window[0]:.0f}\u2013{tnb_window[1]:.0f} nm, {T}%")
    ax_d.axhline(0, color="0.6", lw=0.5)
    ax_d.set_xlabel("Absorbed dose (MGy)")
    ax_d.set_ylabel("\u0394A (DTNB films)")
    ax_d.legend(loc="center right", fontsize=4.6, ncol=1, framealpha=0.9)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_fit_diagnostics(results, out_path, fit_window=(292.0, 470.0),
                           growth_window=(292.0, 340.0),
                           tnb_window=(395.0, 430.0), savgol=(61, 3),
                           calibration=None):
    """Figure S2: every processing step and fit behind the band assignments.

    Built so the fits can be audited rather than taken on trust.  Panel (a)
    overlays raw and smoothed spectra with the residual of smoothing, so the
    effect of the filter is visible.  Panel (b) shows the two-Gaussian
    decomposition with its components and residual.  Panel (c) is the reason
    the fitted band centre cannot be trusted: wavelength-to-wavelength noise
    against wavelength, which rises steeply into the blue where the growth
    band's flank lies.  Panel (d) shows how the fitted DTNB-minus-apo centre
    shift depends on where the fit window starts, next to two model-free
    measures of the same quantity.

    Panels (c) and (d) exist because the fitted shift is not robust: it falls
    from about eight nanometres to under three as the fit window is moved off
    the noisy blue flank, and the model-free measures do not reproduce it.

    Parameters
    ----------
    results
        Processed conditions.
    out_path
        Output path.
    fit_window, growth_window, tnb_window
        Fit range, growth-band search range, and TNB integration window, nm.
    savgol
        ``(window_length, polyorder)`` of the Savitzky-Golay filter applied
        before peak finding.  Fits are performed on filtered spectra; panel (a)
        quantifies what that costs.
    calibration
        Optional ``(angles_deg, lambda_max_nm)`` from an external
        dihedral-to-absorption calibration, annotated on panel (d).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import savgol_filter
    from scipy.optimize import curve_fit

    out_path = Path(out_path)
    pairs = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"), ("DTNB_50", "apo_50")]
    wl = results["DTNB_5"].wavelength_nm
    keep = (wl >= 285.0) & (wl <= 725.0)
    x = wl[keep]

    def gauss(v, A, c, f):
        return A * np.exp(-4 * np.log(2) * ((v - c) / f) ** 2)

    def two_gauss(v, A1, c1, f1, A2, c2, f2):
        return gauss(v, A1, c1, f1) + gauss(v, A2, c2, f2)

    def endpoint(key):
        r = results[key]
        i = int(np.nonzero(r.valid() & (r.dose_MGy <= r.max_valid_dose_MGy()))[0][-1])
        return r, i

    def mean_spectrum(key, smooth=True):
        r, i = endpoint(key)
        raw = np.nanmean(r.delta_a[:, :, i], axis=0)[keep]
        return savgol_filter(raw, *savgol) if smooth else raw

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.subplots_adjust(hspace=0.60, wspace=0.32)

    # --- (a) smoothing: raw, filtered, residual ---
    raw = mean_spectrum("DTNB_25", smooth=False)
    sm = mean_spectrum("DTNB_25", smooth=True)
    ax_a.plot(x, raw, color="0.65", lw=0.5, label="raw")
    ax_a.plot(x, sm, color=SOAK["DTNB"], lw=1.1, label="filtered")
    ax_a.plot(x, raw - sm, color="0.25", lw=0.5, label="residual")
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("\u0394A")
    ax_a.legend(loc="upper right", fontsize=5.2, framealpha=0.9)

    # --- (b) the fit and its components ---
    flo, fhi = fit_window
    fw = (x >= flo) & (x <= fhi)
    xf, yf = x[fw], sm[fw]
    p0 = [yf.max(), 310.0, 50.0, yf.max() * 0.3, 360.0, 80.0]
    bounds = ([0, 295, 20, 0, 330, 30], [1, 330, 90, 1, 430, 200])
    popt, _ = curve_fit(two_gauss, xf, yf, p0=p0, bounds=bounds, maxfev=200000)
    ax_b.plot(xf, yf, color=SOAK["DTNB"], lw=1.1, label="data")
    ax_b.plot(xf, two_gauss(xf, *popt), color="0.2", lw=0.9, ls=(0, (3, 2)),
              label="two-Gaussian fit")
    ax_b.plot(xf, gauss(xf, *popt[:3]), color="0.45", lw=0.7, ls=(0, (1, 1.5)),
              label=f"band 1, {popt[1]:.0f} nm")
    ax_b.plot(xf, gauss(xf, *popt[3:]), color="0.65", lw=0.7, ls=(0, (1, 1.5)),
              label=f"band 2, {popt[4]:.0f} nm")
    ax_b.plot(xf, yf - two_gauss(xf, *popt), color="0.25", lw=0.5,
              label="residual")
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("\u0394A")
    ax_b.legend(loc="upper right", fontsize=4.6, framealpha=0.9)

    # --- (c) noise against wavelength ---
    edges = np.arange(285.0, 726.0, 15.0)
    for dk, ak in pairs[:2]:
        for key, colour in ((dk, SOAK["DTNB"]), (ak, SOAK["apo"])):
            r, i = endpoint(key)
            centres, noise = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                bw = (wl >= lo) & (wl <= hi)
                vals = [np.std(np.diff(r.delta_a[pi, bw, i])) / np.sqrt(2)
                        for pi in range(r.delta_a.shape[0])
                        if np.isfinite(r.delta_a[pi, bw, i]).all()]
                if vals:
                    centres.append(0.5 * (lo + hi))
                    noise.append(np.mean(vals))
            ax_c.plot(centres, noise, color=colour, lw=1.0,
                      alpha=0.55 if dk.endswith("_5") else 1.0)
    shade_band(ax_c, *growth_window)
    shade_band(ax_c, *tnb_window)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Wavelength (nm)")
    ax_c.set_ylabel("Noise (\u0394A per point)")
    key_c = [Line2D([], [], color=SOAK["DTNB"], lw=1.0, label="DTNB"),
             Line2D([], [], color=SOAK["apo"], lw=1.0, label="apo")]
    ax_c.legend(handles=key_c, loc="upper right", fontsize=5.2, framealpha=0.9)

    # --- (d) window sensitivity of the fitted shift ---
    starts = np.arange(292.0, 306.1, 2.0)
    for dk, ak in pairs:
        T = int(results[dk].condition.transmission_pct)
        shifts = []
        for wlo in starts:
            fwv = (x >= wlo) & (x <= fhi)
            cs = []
            for key in (dk, ak):
                yy = mean_spectrum(key)[fwv]
                bb = ([0, max(295, wlo), 20, 0, 330, 30],
                      [1, 330, 90, 1, 430, 200])
                try:
                    pp, _ = curve_fit(two_gauss, x[fwv], yy,
                                      p0=[yy.max(), 310, 50, yy.max() * 0.3, 360, 80],
                                      bounds=bb, maxfev=200000)
                    cs.append(pp[1])
                except Exception:
                    cs.append(np.nan)
            shifts.append(cs[0] - cs[1])
        ax_d.plot(starts, shifts, color=TRANSMISSION[T], lw=1.1,
                  marker="o", ms=2.2, label=f"{T}%")
    ax_d.axhline(0, color="0.4", lw=0.6)
    ax_d.set_xlabel("Fit window start (nm)")
    ax_d.set_ylabel("Fitted DTNB \u2212 apo\ncentre shift (nm)")
    ax_d.legend(loc="lower right", fontsize=5.2, ncol=3, framealpha=0.9,
                title="transmission", title_fontsize=5.2)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


DIAGNOSTIC_NM = {310.0: "growth band",
                 328.0: "protein\u2013S\u2013S\u2013TNB",
                 400.0: "disulphide radical",
                 412.0: "free TNB$^{2-}$",
                 480.0: "isosbestic",
                 580.0: "solvated e$^-$"}


# Room-temperature diffraction lifetimes (MGy) as a function of dose rate,
# from Owen et al. 2012; the inverse dose-rate effect means the usable dose
# depends on how fast it is delivered.
RT_LIFETIME_BY_DOSE_RATE = {0.261: 0.257, 0.521: 0.300, 1.042: 0.373}

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
        ax_b.annotate(f"{r['T']}%", xy=(r["rate"], ratio), xytext=(5, 0),
                      textcoords="offset points", fontsize=5.4,
                      color=TRANSMISSION[r["T"]], va="center")
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
    ax_b.axhline(1.0, color="0.3", lw=0.7)
    ax_b.annotate("feasible on one crystal below here",
                  xy=(ax_b.get_xlim()[1], 1.0), xytext=(0, -6),
                  textcoords="offset points", fontsize=5.0, color="0.3",
                  ha="right", va="top")
    ax_b.set_xscale("log")
    ax_b.set_ylim(0, None)
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


def figure_buffer_controls(results, buffer_static, buffer_irradiated, out_path,
                           condition="DTNB_25", control="apo_25",
                           windows=((300.0, 325.0), (395.0, 430.0), (600.0, 700.0)),
                           wl_range=(295.0, 700.0), scatter_window=(470.0, 725.0),
                           scatter_exponent=3.0):
    """Figure S3: mother-liquor-only controls, on a time axis.

    Two buffer acquisitions from the visit directory bound what the mother liquor
    contributes.  The static one establishes how much of the raw absorbance is
    not protein; the irradiated one establishes whether the liquor generates any
    radiation-driven signal of its own.

    The x-axis here is elapsed time, not dose, because no transmission was
    recorded for the buffer acquisition -- its header gives only a 2 s kinetic
    cycle.  Dose cannot be assigned to it, so a time axis is the honest choice
    and the comparison to the film traces is qualitative.

    Parameters
    ----------
    results
        Processed conditions, for the film comparison in panel (a).
    buffer_static
        ``Acquisition`` of the single-frame buffer spectrum.
    buffer_irradiated
        ``Acquisition`` of the irradiated buffer kinetics series.
    out_path
        Output path.
    condition, control
        Film conditions to compare against: the DTNB-soaked arm and its matched
        unlabelled control, both shown so the liquor contribution can be judged
        against each.
    windows
        Integration windows for the time traces in panel (c).
    wl_range
        Wavelength limits.
    scatter_window, scatter_exponent
        Passed to the power-law scatter model.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    wl_lo, wl_hi = wl_range

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.46, bottom=0.24)

    wb = buffer_static.wavelength_nm
    ab = buffer_static.absorbance[:, 0]
    keep = (wb >= wl_lo) & (wb <= wl_hi)

    # --- (a) absolute ground state: buffer against both film arms ---
    # Pre-irradiation (dark) spectra, so this is the sample as loaded.  Both arms
    # are shown because the liquor contributes to each and the reader needs to see
    # that the DTNB-apo difference is not itself a liquor artefact.
    for key, colour in ((condition, SOAK["DTNB"]), (control, SOAK["apo"])):
        rr = results[key]
        film = np.nanmean(np.stack([t.raw_dark for t in rr.traces]), axis=0)
        T = rr.condition.transmission_pct
        soak = SOAK_LABEL.get(key.split("_")[0], key.split("_")[0])
        ax_a.plot(wb[keep], film[keep], color=colour, lw=1.2,
                  label=f"{soak}, {T:.0f}% T (n={len(rr.traces)})")
    ax_a.plot(wb[keep], ab[keep], color="0.45", lw=1.2,
              label="mother liquor only")
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("Absorbance, pre-irradiation")
    ax_a.legend(loc="upper right", fontsize=4.6, framealpha=0.0,
                handlelength=1.2, labelspacing=0.22)

    # --- (b) scatter-corrected buffer: is it featureless? ---
    sc = remove_scatter(wb, ab[:, None], exponent=scatter_exponent,
                        window=scatter_window)[:, 0]
    model = ab - sc
    ax_b.plot(wb[keep], ab[keep], color="0.55", lw=1.0, label="liquor, measured")
    ax_b.plot(wb[keep], model[keep], color="0.30", lw=0.9, ls=(0, (3, 2)),
              label=(f"fitted $\\lambda^{{-{scatter_exponent:g}}}$ scatter, "
                     f"{scatter_window[0]:.0f}\u2013{scatter_window[1]:.0f} nm"))
    # Greyscale only: the soak-state colours are reserved for DTNB and apo, and
    # this residual is a liquor scatter term with no connection to either arm.
    ax_b.plot(wb[keep], sc[keep], color="0.05", lw=1.0,
              label="residual (measured \u2212 fit)")
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("Wavelength (nm)")
    ax_b.set_ylabel("Absorbance")
    ax_b.legend(loc="upper right", fontsize=4.6, framealpha=0.0,
                handlelength=1.4, labelspacing=0.22)

    # --- (c) irradiated buffer, time axis ---
    wi = buffer_irradiated.wavelength_nm
    Ai = buffer_irradiated.absorbance
    ti = buffer_irradiated.time_s()
    # The buffer acquisition aborted partway through: from one frame onward every
    # raw value is exactly zero and the file pads the remaining frames with
    # duplicates.  Those are not measurements, so the series is cut at the last
    # frame carrying real counts rather than plotted as a flat line.
    real = np.nanmax(np.abs(Ai), axis=0) > 0
    if real.any():
        last = int(np.nonzero(real)[0][-1]) + 1
        Ai, ti = Ai[:, :last], ti[:last]
    dark = np.nanmean(Ai[:, :5], axis=1)
    corr = remove_scatter(wi, Ai - dark[:, None], exponent=scatter_exponent,
                          window=scatter_window)
    greys = ["0.15", "0.45", "0.7"]
    for (lo, hi), col in zip(windows, greys):
        m = (wi >= lo) & (wi <= hi)
        ax_c.plot(ti, np.nanmean(corr[m, :], axis=0), color=col, lw=1.1,
                  label=f"{lo:.0f}\u2013{hi:.0f} nm")
    ax_c.axhline(0, color="0.6", lw=0.5)
    ax_c.set_xlabel("Elapsed time (s)")
    # Kept short: a two-line label at this panel width pushes into panel (b).
    # What the delta is referenced to is stated in the caption instead.
    ax_c.set_ylabel("\u0394A, liquor (vs frames 1\u20135)")
    ax_c.legend(loc="lower left", fontsize=5.2, framealpha=0.0,
                handlelength=1.3, labelspacing=0.25)

    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig


def figure_shoulder_isosbestic(results, buffer_static, raw_dir, out_path,
                               condition="DTNB_25", control="apo_25",
                               savgol=(61, 3), band=(415.0, 455.0),
                               cross_range=(380.0, 560.0), n_boot=4000, seed=11):
    """Figure S4: is the 400-500 nm region a shoulder, an isosbestic point, or neither?

    Two distinct questions, tested separately.

    A *shoulder* is a curvature question: a band or shoulder shows negative second
    derivative.  Testing the arms separately is not enough, because the
    spectrograph's own response puts negative curvature near its 454 nm centre
    wavelength in labelled, unlabelled and mother-liquor spectra alike.  The
    label-specific quantity is the curvature of the DTNB-minus-apo difference,
    which subtracts any feature common to both.

    An *isosbestic point* is a dose-invariance question: a wavelength where two
    interconverting species have equal extinction shows zero change at every
    dose.  It must be tested on raw dark-referenced data, because inside the
    scatter-fit window the correction drives the difference toward zero by
    construction, which would manufacture a spurious one.

    Parameters
    ----------
    results
        Processed conditions.
    buffer_static
        Single-frame mother-liquor ``Acquisition``, as the instrument-response arm.
    raw_dir
        Directory of raw ``.asc`` files, re-read to get uncorrected traces.
    out_path
        Output path.
    condition, control
        Labelled and matched unlabelled condition keys.
    savgol
        ``(window, polyorder)`` for the derivative estimate.
    band
        Window over which the excess curvature is summarised and bootstrapped.
    cross_range
        Wavelength range searched for zero crossings.
    n_boot, seed
        Bootstrap over film positions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import savgol_filter

    out_path = Path(out_path)
    raw_dir = Path(raw_dir)
    rd, ra = results[condition], results[control]
    w = rd.wavelength_nm
    b_lo, b_hi = band

    gd = np.stack([t.raw_dark for t in rd.traces])
    ga = np.stack([t.raw_dark for t in ra.traces])
    gs_d, gs_a = np.nanmean(gd, axis=0), np.nanmean(ga, axis=0)
    buf = np.interp(w, buffer_static.wavelength_nm,
                    buffer_static.absorbance[:, 0])

    def second_deriv(y):
        return np.gradient(np.gradient(savgol_filter(y, *savgol), w), w)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.46, bottom=0.24)

    keep = (w >= 360) & (w <= 560)
    m = (w >= b_lo) & (w <= b_hi)

    # --- (a) the spectra themselves, normalised so shape can be compared ---
    # The raw second derivative is uninterpretable at full spectral resolution:
    # pixel-to-pixel noise reaches +/-0.002 against a candidate feature of
    # 1e-05, and it is larger in the apo arm.  What the eye needs here is the
    # shape of each spectrum over the region in question; the quantitative test
    # lives in panel (b).
    ref = (w >= 470.0) & (w <= 500.0)
    for y, colour, lab in ((gs_d, SOAK["DTNB"], "DTNB-soaked"),
                           (gs_a, SOAK["apo"], "apo"),
                           (buf, "0.55", "mother liquor")):
        norm = y / np.nanmean(y[ref])
        ax_a.plot(w[keep], norm[keep], color=colour, lw=1.0, label=lab)
    ax_a.axvspan(b_lo, b_hi, color="0.90", zorder=0)
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("Absorbance / mean(470\u2013500 nm)")
    ax_a.legend(loc="upper right", fontsize=4.6, framealpha=0.0,
                handlelength=1.2, labelspacing=0.22)

    # --- (b) is the DTNB-apo curvature bigger than a within-arm split? ---
    # A position bootstrap is the WRONG null here: it estimates the precision of
    # the mean, which shrinks with n, and so reports a confidence interval
    # excluding zero for a difference no larger than the spread between two
    # arbitrary halves of a single arm.  The null drawn here splits each arm in
    # half and differences the halves, which is what "no label effect" looks like.
    rng = np.random.default_rng(seed)

    def split_diff(stack):
        idx = rng.permutation(len(stack))
        h = len(stack) // 2
        return (np.nanmean(stack[idx[:h]], axis=0)
                - np.nanmean(stack[idx[h:2 * h]], axis=0))

    null = np.array([np.nanmean(second_deriv(split_diff(st))[m])
                     for st in (gd, ga) for _ in range(n_boot // 2)])
    obs = float(np.nanmean(second_deriv(gs_d - gs_a)[m]))
    p_null = float((null <= obs).mean())

    ax_b.hist(null, bins=40, color="0.75", edgecolor="none")
    ax_b.axvline(obs, color="0.05", lw=1.2,
                 label=f"observed DTNB \u2212 apo\n{obs:+.1e}, p = {p_null:.2f}")
    ax_b.set_xlabel(f"d$^2$/d$\\lambda^2$ over {b_lo:.0f}\u2013{b_hi:.0f} nm")
    ax_b.set_ylabel("Within-arm split draws")
    ax_b.legend(loc="upper left", fontsize=4.4, framealpha=0.0,
                handlelength=1.0, labelspacing=0.22)

    # --- (c) zero crossing of RAW dA against dose: does it stay put? ---
    raws = []
    for t in rd.traces:
        A = read_asc(raw_dir / t.filename).absorbance
        raws.append(A - np.nanmean(A[:, :5], axis=1)[:, None])
    R = np.nanmean(np.stack(raws), axis=0)

    v = rd.valid() & (rd.dose_MGy <= rd.max_valid_dose_MGy())
    nf = min(R.shape[1], int(np.nonzero(v)[0][-1]) + 1)
    cb = (w >= cross_range[0]) & (w <= cross_range[1])

    frames = np.linspace(0, nf - 1, 60).astype(int)
    xs, ys = [], []
    for f in frames:
        sgn = np.sign(R[cb, f])
        cr = w[cb][:-1][np.diff(sgn) != 0]
        if len(cr):
            xs.append(rd.dose_MGy[f])
            ys.append(float(np.nanmedian(cr)))
    ax_c.plot(xs, ys, color="0.05", lw=0.9, marker="o", ms=1.8, mew=0)
    ax_c.set_xlabel("Absorbed dose (MGy)")
    ax_c.set_ylabel("Zero crossing of raw \u0394A (nm)")

    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig
