"""Thesis figures, each built from processed data in a single function.

Every figure takes the processed :class:`~crystalspec.pipeline.ConditionResult`
objects and writes one file.  No figure reads a CSV written by another stage,
so there is exactly one path from the raw ``.asc`` files to any panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
)

__all__ = ["figure_qc", "figure_signature", "figure_cryo", "figure_dose_dependence"]


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
    axa.set_title("Raw spectra scale with material in beam")
    axa.text(0.96, 0.94, "DTNB, 5% T\n5 film positions", transform=axa.transAxes,
             ha="right", va="top", fontsize=6)

    # (b) after scatter removal
    for acq, col in zip(acqs, shades):
        axb.plot(wl[vis], remove_scatter(wl, acq.absorbance[:, 0])[vis], lw=0.7, color=col)
    shade_band(axb, 470, 725)
    axb.axhline(0, color="0.55", lw=0.5)
    axb.set_xlabel("Wavelength (nm)")
    axb.set_ylabel("Absorbance")
    axb.set_title("Red tail flattened by scatter removal")
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
    axc.set_title("Scattering varies severalfold between positions")

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
    axd.set_title("Resets detected as broadband jumps")
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
    axe.set_title("Isolated resets stitched, burst region cut")
    axe.legend(loc="lower right")

    # (f) per-position difference traces
    res = results[demo_key]
    for i, col in zip(range(res.n_crystals), shades):
        axf.plot(res.dose_MGy, res.delta_a[i, i412, :], lw=0.7, color=col)
    axf.axhline(0, color="0.55", lw=0.5)
    axf.set_xlabel("Dose (MGy)")
    axf.set_ylabel("$\\Delta A_{412}$")
    axf.set_title("Referencing reveals a common response")

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
    axg.set_title("Usable window per film position", pad=14)
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
    ax.set_title("100 K, after − before X-rays")
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
    ax.set_title("Room temperature, 5% T")
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
    ax.set_title("TNB window: 100 K resists")
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


def figure_dose_dependence(results, cross_validation, out_path,
                           tnb_window=(395.0, 430.0), n_boot=2000, seed=0):
    """Figure 2: dose dependence of the TNB signal.

    Panel (a) shows the TNB-window difference signal against absorbed dose
    for the four attenuator settings, DTNB above apo.  Panel (b) shows the
    per-position signal integrated over each pair's common dose range, the
    quantity the significance test is computed on, so the reader sees the
    five values behind each p.  Panel (c) compares the characteristic dose
    recovered independently by band integration and by singular value
    decomposition.

    Parameters
    ----------
    results
        Processed conditions.
    cross_validation
        DataFrame from the cross-validation table, one row per transmission
        level, carrying the fitted doses and test results.
    out_path
        Output path.
    tnb_window
        Integration window, nm.
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
                          hspace=0.62, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- (a) TNB signal vs dose, all levels ---
    for dk, ak in pairs:
        d, a = results[dk], results[ak]
        T = int(d.condition.transmission_pct)
        common = min(d.max_valid_dose_MGy(), a.max_valid_dose_MGy())
        for res, style in ((d, "-"), (a, ":")):
            bs = band_series(res, tnb_window, "TNB", n_boot=n_boot, seed=seed)
            keep = (bs.dose_MGy <= common) & (bs.dose_MGy > 0)
            ax_a.plot(bs.dose_MGy[keep], bs.mean[keep], style,
                      color=TRANSMISSION[T], lw=1.2 if style == "-" else 0.9)
            if style == "-":
                ax_a.fill_between(bs.dose_MGy[keep], bs.lo[keep], bs.hi[keep],
                                  color=TRANSMISSION[T], alpha=0.16, lw=0)
                ax_a.annotate(f"{T}% T", xy=(bs.dose_MGy[keep][-1], bs.mean[keep][-1]),
                              xytext=(3, 0), textcoords="offset points",
                              color=TRANSMISSION[T], fontsize=6, va="center")
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.02, 45)
    ax_a.set_xticks([0.1, 1, 10])
    ax_a.set_xticklabels(["0.1", "1", "10"])
    ax_a.set_xlabel("Absorbed dose (MGy)")
    ax_a.set_ylabel("ΔA, 395–430 nm")
    ax_a.set_title("DTNB films lose TNB-window absorbance at every dose rate")
    ax_a.annotate("solid: DTNB      dotted: apo", xy=(0.015, 0.08),
                  xycoords="axes fraction", fontsize=6, color="0.3")

    # --- (b) per-position AUC, the test statistic ---
    rng = np.random.default_rng(seed)
    for i, (dk, ak) in enumerate(pairs):
        d, a = results[dk], results[ak]
        common = min(d.max_valid_dose_MGy(), a.max_valid_dose_MGy())
        for off, (res, soak) in zip((-0.16, 0.16), ((d, "DTNB"), (a, "apo"))):
            valid = res.valid() & (res.dose_MGy <= common)
            per = integrate_band(res.wavelength_nm, res.delta_a[:, :, valid], tnb_window)
            dose = res.dose_MGy[valid]
            aucs = np.array([
                np.trapezoid(p[np.isfinite(p)], dose[np.isfinite(p)]) for p in per
            ])
            ax_b.scatter(np.full(aucs.size, i + off) + rng.uniform(-0.05, 0.05, aucs.size),
                         aucs, s=11, lw=0, color=SOAK[soak], alpha=0.9)
            ax_b.plot([i + off - 0.10, i + off + 0.10], [aucs.mean()] * 2,
                      color="0.2", lw=1.1)
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_yscale("symlog", linthresh=0.1)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels([f"{int(results[d].condition.transmission_pct)}%" for d, _ in pairs])
    ax_b.set_xlabel("X-ray transmission")
    ax_b.set_ylabel("∫ΔA d(dose)")
    ax_b.set_title("Per-position test statistic")
    for i, (_, row) in enumerate(cross_validation.iterrows()):
        mark = "*" if row["band_p_bonferroni"] < 0.05 else "n.s."
        ax_b.annotate(mark, xy=(i, ax_b.get_ylim()[1]), xytext=(0, -6),
                      textcoords="offset points", ha="center", va="top",
                      fontsize=7 if mark == "*" else 5.5, color="0.25")

    # --- (c) characteristic dose, two independent routes ---
    T = cross_validation["transmission_pct"].to_numpy()
    band_d = cross_validation["band_halfdose_MGy"].to_numpy()
    svd_d = cross_validation["svd_D1_MGy"].to_numpy()
    band_lo = np.array([float(s.strip("[]").split(",")[0])
                        for s in cross_validation["band_halfdose_ci"]])
    band_hi = np.array([float(s.strip("[]").split(",")[1])
                        for s in cross_validation["band_halfdose_ci"]])
    svd_lo = np.array([float(s.strip("[]").split(",")[0])
                       for s in cross_validation["svd_D1_ci"]])
    svd_hi = np.array([float(s.strip("[]").split(",")[1])
                       for s in cross_validation["svd_D1_ci"]])
    ax_c.errorbar(T, band_d, yerr=[band_d - band_lo, band_hi - band_d],
                  fmt="o-", ms=4, lw=1.0, capsize=2, elinewidth=0.7,
                  color="#1b3a6b", label="band integration")
    ax_c.errorbar(T * 1.08, svd_d, yerr=[svd_d - svd_lo, svd_hi - svd_d],
                  fmt="s--", ms=4, lw=1.0, capsize=2, elinewidth=0.7,
                  color="#7a9cc6", label="SVD component")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xticks([5, 25, 50, 100])
    ax_c.set_xticklabels(["5", "25", "50", "100"])
    ax_c.set_xlabel("X-ray transmission (%)")
    ax_c.set_ylabel("Characteristic dose (MGy)")
    ax_c.set_title("Characteristic dose rises with dose rate")
    ax_c.legend(loc="upper left", fontsize=5.5)

    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, letter, dx=-0.10 if ax is ax_a else -0.30)

    fig.savefig(out_path)
    overlaps = check_overlaps(fig)
    if overlaps:
        import warnings
        warnings.warn(f"{out_path.name}: {len(overlaps)} overlaps: {overlaps[:4]}",
                      stacklevel=2)
    return fig
