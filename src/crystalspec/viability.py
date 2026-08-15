"""Three figures and one table that answer a single question.

    Can the loss of a DTNB label be followed spectroscopically inside the dose
    a room-temperature crystal survives?

This module deliberately does NOT re-derive anything.  It reads the quantities
the main pipeline already computed and presents them.  Every number it plots
comes from ``table_10_dose_budget.csv`` (detection doses, dose rates, published
diffraction lifetimes) or from the processed traces in the pipeline result.

WHAT THE THREE FIGURES SHOW

V1  The signal is real and needs X-rays.  The label band at 412 nm falls under
    irradiation; over the same interval a soaked film with the shutter closed
    stays inside the baseline noise (-0.0016 at 20 s against a 0.004 floor).
    The control is NOT flat over its full 100 s record -- it drifts to -0.027,
    a peak-to-peak excursion of 0.028, roughly 7x the noise floor and
    comparable to the label loss itself.  That drift is why the comparison is
    bounded at t_max_s and read at a matched time: the control is evidence
    only on the timescale the irradiated films are analysed over.

V2  The verdict.  Detection dose against published room-temperature
    diffraction lifetimes, for each dose rate.  Below the diagonal the label
    change is measurable before the crystal dies.

V3  The catch.  What else changes at the same time, and by how much -- the
    protein's own radiation chemistry is the larger signal and sets the
    contrast limit, not the detector.

WHAT THESE FIGURES DO NOT CLAIM

  * No concentration, occupancy or extinction coefficient.  The films contain
    an unknown and variable crystalline volume fraction plus mother liquor, so
    only relative changes are interpretable.
  * No mechanism.  Whether the chromophore is destroyed in place or released
    and lost from the probed volume is not decidable from these data.
  * No rate constant offered as a property of the chemistry.  A fitted decay
    constant here depends on the dose window chosen (it grows 1.4-2.5x between
    a 0.6 and a 1.5 MGy window), so the decay is not single-exponential and any
    single number would be an artefact of the window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import SOAK, TRANSMISSION, apply_style, check_overlaps, panel_label

META_GREY = "0.6"

TNB_WINDOW_NM = (408.0, 416.0)
GROWTH_NM = 310.0
NOISE_FLOOR_DA = 0.004
NO_XRAY_FILE = "hgd_R37S_DTNB_noxrays.asc"

__all__ = ["figure_v1_signal_is_real", "figure_v2_verdict",
           "figure_v3_what_else_changes", "table_viability"]


def _band(result, lo, hi, positions=None):
    """Mean delta_a over a wavelength band, positions held fixed.

    The contributing set of film positions is FIXED at the three that survive
    longest.  Averaging over a shrinking set puts a step in the mean when a
    position drops out -- an artefact, not a spectrometer reset.
    """
    b = (result.wavelength_nm >= lo) & (result.wavelength_nm <= hi)
    v = result.valid() & (result.dose_MGy <= result.max_valid_dose_MGy())
    finite = np.isfinite(result.delta_a[:, 0, :])
    last = np.array([np.nonzero(r)[0][-1] if r.any() else -1 for r in finite])
    keep = np.argsort(last)[::-1][:3] if positions is None else positions
    with np.errstate(invalid="ignore"):
        y = np.nanmean(np.nanmean(result.delta_a[keep][:, b, :][:, :, v], axis=1), axis=0)
    return result.dose_MGy[v], result.time_s[v], y


def _difference(results, dtnb_key, apo_key, lo, hi):
    """DTNB minus the matched apo control, on the DTNB dose axis.

    The subtraction is not cosmetic.  The protein's own radiation chemistry
    produces a band near 310 nm whose red tail reaches 412 nm and lifts the
    single-arm trace back up after it falls; the apo arm carries that growth
    without any label.  Scaled by one number it reproduces the single-arm
    'recovery' to within 1 %, so subtracting it leaves a monotonic trace that
    needs no truncation.
    """
    dd, td, yd = _band(results[dtnb_key], lo, hi)
    da, _, ya = _band(results[apo_key], lo, hi)
    return dd, td, yd - np.interp(dd, da, ya)


def _no_xray_control(registry):
    """The soaked film with the shutter closed, same processing chain.

    Loaded through the registry so the file is resolved the same way as every
    other acquisition, and processed with the same scatter removal, so that a
    flat result here is evidence about the sample rather than about a
    differently-treated control.
    """
    from .io import read_asc
    from .scatter import remove_scatter

    acq = read_asc(Path(registry.raw_dir) / NO_XRAY_FILE)
    w, a = acq.wavelength_nm, acq.absorbance
    a = a[:, np.abs(a).sum(axis=0) > 0]          # header over-reports frames
    d = a - a[:, :5].mean(axis=1, keepdims=True)
    dc = np.column_stack([remove_scatter(w, d[:, i]) for i in range(d.shape[1])])
    b = (w >= TNB_WINDOW_NM[0]) & (w <= TNB_WINDOW_NM[1])
    return np.arange(dc.shape[1]) * 0.1, dc[b].mean(axis=0)


def figure_v1_signal_is_real(results, pairs, registry, out_path, t_max_s=20.0):
    """V1: the label band falls under X-rays; without them it stays in the noise.

    Plotted against TIME, not dose, because the control has no dose axis --
    the shutter was never opened.  t_max_s bounds the panel at the point where
    the fastest condition has run out of usable frames.

    t_max_s is load-bearing, not cosmetic.  The no-X-ray control is flat only
    over the first ~20 s; by 100 s it has drifted to -0.027, which is
    comparable to the label loss.  Extending this panel to the control's full
    record would show a 'control' with an excursion as large as the effect.
    The defensible statement is therefore bounded: over the interval the
    irradiated films are analysed, the unirradiated film stays within the
    baseline noise while the irradiated ones fall 4-7x below it.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    for dk, ak in pairs:
        t_pct = int(dk.split("_")[1])
        _, td, yd = _difference(results, dk, ak, *TNB_WINDOW_NM)
        m = td - td[0] <= t_max_s
        axa.plot(td[m] - td[0], yd[m], lw=1.2, color=TRANSMISSION[t_pct],
                 label=f"{t_pct} % transmission")
    tc, yc = _no_xray_control(registry)
    mc = tc <= t_max_s
    axa.plot(tc[mc], yc[mc], lw=1.4, color="0.25", ls="--", label="no X-rays")
    axa.axhline(0, color=META_GREY, lw=0.5)
    axa.set_xlabel("Time since shutter opening (s)")
    axa.set_ylabel("Label signal, $\\Delta A$ at 412 nm")
    axa.legend(loc="lower right", frameon=False, handlelength=1.6,
               labelspacing=0.25, borderpad=0.3)

    # Panel b: one number per condition at a MATCHED time.  The high-dose-rate
    # films fail optically within a few seconds, so a bar chart read at each
    # condition's own last frame would silently compare 3 s against 20 s.
    # t_match is the largest time every condition still has data for.
    ends = []
    for dk, ak in pairs:
        _, td, _ = _difference(results, dk, ak, *TNB_WINDOW_NM)
        ends.append(float(td[-1] - td[0]))
    t_match = min(ends)
    labels, vals = [], []
    for dk, ak in pairs:
        _, td, yd = _difference(results, dk, ak, *TNB_WINDOW_NM)
        labels.append(f"{int(dk.split('_')[1])} %")
        vals.append(float(np.interp(t_match, td - td[0], yd)))
    labels.append("no X-rays")
    vals.append(float(np.interp(t_match, tc, yc)))
    colours = [TRANSMISSION[int(dk.split("_")[1])] for dk, _ in pairs] + ["0.25"]
    bars = axb.bar(labels, vals, color=colours, width=0.62)
    # The control bar is near zero by construction; annotate it so "no change"
    # reads as a measured result rather than a missing bar.
    axb.annotate(f"{vals[-1]:+.4f}", (bars[-1].get_x() + bars[-1].get_width() / 2,
                                      vals[-1]), xytext=(0, -4),
                 textcoords="offset points", ha="center", va="top",
                 fontsize=6, color="0.25")
    axb.axhspan(-NOISE_FLOOR_DA, NOISE_FLOOR_DA, color=META_GREY, alpha=0.22, lw=0)
    axb.axhline(0, color=META_GREY, lw=0.5)
    axb.annotate("baseline noise", (len(labels) - 0.45, NOISE_FLOOR_DA),
                 xytext=(0, 3), textcoords="offset points", ha="right",
                 va="bottom", fontsize=6, color="0.35")
    axb.set_ylabel(f"$\\Delta A$ at 412 nm after {t_match:.1f} s")
    axb.tick_params(axis="x", rotation=0)

    for ax, letter in ((axa, "a"), (axb, "b")):
        panel_label(ax, letter)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    return fig


def figure_v2_verdict(results, pairs, budget_rows, out_path, dose_max_MGy=0.4):
    """V2: dose against the label signal, with the detection threshold marked.

    This plots the quantity a reader actually wants to see -- how big is the
    change, at what dose, relative to the noise -- rather than a derived
    ratio.  ``budget_rows`` supplies the pre-computed detection dose and
    published lifetimes so the two panels cannot drift from table 10.

    Panel a: |label signal| (the DTNB-apo difference at 412 nm) against dose,
    one curve per transmission, with the measured noise floor shaded and each
    curve's detection dose marked where it crosses that floor.  The vertical
    band is the published room-temperature diffraction-lifetime range
    (Owen et al. 2012); a detection mark to the LEFT of the band means the
    label change is measurable before a comparable crystal stops diffracting.

    Panel b: the same detection dose against dose rate, so the trend -- slower
    delivery detects at lower dose -- is visible directly rather than folded
    into a ratio.  The same lifetime band is repeated on the y-axis.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    by_cond = {}
    for r in budget_rows:
        by_cond.setdefault(r["condition"], []).append(r)
    lo_life = min(float(r["diffraction_lifetime_MGy"]) for r in budget_rows)
    hi_life = max(float(r["diffraction_lifetime_MGy"]) for r in budget_rows)

    axa.axvspan(lo_life, hi_life, color=META_GREY, alpha=0.18, lw=0)
    axa.annotate("published RT\ncrystal lifetime", (0.5 * (lo_life + hi_life), 0.030),
                 ha="center", va="top", fontsize=6, color="0.35")
    axa.axhspan(0, NOISE_FLOOR_DA, color=META_GREY, alpha=0.30, lw=0)
    axa.annotate("baseline noise", (dose_max_MGy * 0.98, NOISE_FLOOR_DA),
                 xytext=(0, 2), textcoords="offset points", ha="right",
                 va="bottom", fontsize=6, color="0.35")

    for dkey, akey in pairs:
        t_pct = int(dkey.split("_")[1])
        _, _, lab = _difference(results, dkey, akey, *TNB_WINDOW_NM)
        d, _, _ = _difference(results, dkey, akey, *TNB_WINDOW_NM)
        m = d <= dose_max_MGy
        y = np.abs(lab[m])
        axa.plot(d[m], y, lw=1.3, color=TRANSMISSION[t_pct], label=f"{t_pct} %")
        det = float(by_cond[dkey][0]["dose_to_clear_noise_MGy"])
        axa.plot([det], [NOISE_FLOOR_DA], "o", ms=3.6, color=TRANSMISSION[t_pct],
                 zorder=5)
    axa.set_xlim(0, dose_max_MGy)
    axa.set_ylim(0, 0.032)
    axa.set_xlabel("Absorbed dose (MGy)")
    axa.set_ylabel("Label signal, $|\\Delta A|$ at 412 nm")
    axa.legend(loc="center right", frameon=False, handlelength=1.6,
               labelspacing=0.22, borderpad=0.3, fontsize=6)

    # Panel b: detection dose against dose rate -- the design curve.  Slower
    # delivery detects the label change at lower dose, which is the
    # counter-intuitive result worth showing on its own axis.
    axb.axhspan(lo_life, hi_life, color=META_GREY, alpha=0.18, lw=0)
    axb.annotate("published RT\ncrystal lifetime", (0.06, hi_life), xytext=(0, 2),
                 textcoords="offset points", ha="left", va="bottom",
                 fontsize=6, color="0.35")
    for cond, rows in by_cond.items():
        t_pct = int(cond.split("_")[1])
        rate = float(rows[0]["dose_rate_MGy_per_s"])
        det = float(rows[0]["dose_to_clear_noise_MGy"])
        axb.plot([rate], [det], "o", ms=4.2, color=TRANSMISSION[t_pct])
        axb.annotate(f"{t_pct} %", (rate, det), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=6, color=TRANSMISSION[t_pct])
    rates = sorted(float(by_cond[c][0]["dose_rate_MGy_per_s"]) for c in by_cond)
    dets = [float(by_cond[c][0]["dose_to_clear_noise_MGy"]) for c in
            sorted(by_cond, key=lambda k: float(by_cond[k][0]["dose_rate_MGy_per_s"]))]
    axb.plot(rates, dets, lw=0.8, color=META_GREY, ls=":", zorder=0)
    axb.set_xscale("log")
    axb.set_ylim(0, hi_life * 1.15)
    axb.set_xlabel("Dose rate (MGy s$^{-1}$)")
    axb.set_ylabel("Dose to detect the label change (MGy)")

    for ax, letter in ((axa, "a"), (axb, "b")):
        panel_label(ax, letter)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    return fig


def figure_v3_what_else_changes(results, pairs, out_path, dose_max_MGy=1.0):
    """V3: the protein's own chemistry is the bigger signal.

    Both arms grow a band near 310 nm under irradiation.  It is several times
    larger than the label loss and it overlies 328 nm, where the mixed
    protein-S-S-TNB disulphide absorbs -- which is why the label has to be read
    at 412 nm as a difference, and why the intact adduct cannot be followed at
    all.  ``dose_max_MGy`` bounds the axis to the range every condition reaches.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    dk, ak = pairs[1]                      # 25 % transmission: longest record
    for key, colour, label in ((dk, SOAK["DTNB"], "DTNB-soaked"),
                               (ak, SOAK["apo"], "apo control")):
        d, _, y310 = _band(results[key], GROWTH_NM - 4, GROWTH_NM + 4)
        d4, _, y412 = _band(results[key], *TNB_WINDOW_NM)
        m, m4 = d <= dose_max_MGy, d4 <= dose_max_MGy
        axa.plot(d[m], y310[m], lw=1.3, color=colour, label=f"{label}, 310 nm")
        axa.plot(d4[m4], y412[m4], lw=1.3, color=colour, ls=":",
                 label=f"{label}, 412 nm")
    axa.axhline(0, color=META_GREY, lw=0.5)
    axa.set_xlabel("Absorbed dose (MGy)")
    axa.set_ylabel("$\\Delta A$")
    axa.legend(loc="upper left", frameon=False, handlelength=1.8,
               labelspacing=0.22, borderpad=0.3, fontsize=5.6)

    # Panel b: ratio of the two, per condition -- how much bigger the
    # background chemistry is than the thing being measured.
    labels, ratios, colours = [], [], []
    for dkey, akey in pairs:
        t_pct = int(dkey.split("_")[1])
        d, _, g = _band(results[dkey], GROWTH_NM - 4, GROWTH_NM + 4)
        _, _, lab = _difference(results, dkey, akey, *TNB_WINDOW_NM)
        m = d <= dose_max_MGy
        n = min(m.sum(), len(lab))
        if n < 5 or abs(lab[:n]).max() < 1e-9:
            continue
        labels.append(f"{t_pct} %")
        ratios.append(float(abs(g[m][:n]).max() / abs(lab[:n]).max()))
        colours.append(TRANSMISSION[t_pct])
    axb.bar(labels, ratios, color=colours, width=0.6)
    axb.axhline(1.0, color=META_GREY, lw=0.8, ls="--")
    axb.annotate("equal size", (len(labels) - 0.45, 1.0), xytext=(0, 3),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6, color="0.35")
    axb.set_ylabel("Protein growth / label loss")
    axb.set_xlabel("Transmission")

    for ax, letter in ((axa, "a"), (axb, "b")):
        panel_label(ax, letter)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    return fig


def table_viability(results, pairs, budget_rows, out_path, dose_max_MGy=1.0):
    """One row per condition: everything the viability claim rests on.

    Columns are deliberately few and each is directly measured or directly
    read from the dose budget.  No fitted rate constants appear, because a
    fitted decay constant on these traces depends on the dose window chosen.
    """
    import csv

    by_cond = {}
    for r in budget_rows:
        by_cond.setdefault(r["condition"], []).append(r)

    rows = []
    for dkey, akey in pairs:
        t_pct = int(dkey.split("_")[1])
        b = by_cond[dkey]
        d, _, lab = _difference(results, dkey, akey, *TNB_WINDOW_NM)
        dg, _, g = _band(results[dkey], GROWTH_NM - 4, GROWTH_NM + 4)
        m = d <= dose_max_MGy
        mg = dg <= dose_max_MGy
        ratios = [float(r["detection_dose_over_lifetime"]) for r in b]
        rows.append({
            "transmission_pct": t_pct,
            "dose_rate_MGy_per_s": float(b[0]["dose_rate_MGy_per_s"]),
            "max_usable_dose_MGy": round(float(d[-1]), 2),
            "label_change_by_1MGy_dA": round(float(lab[m][-1]), 4),
            "baseline_noise_dA": NOISE_FLOOR_DA,
            "detection_dose_MGy": float(b[0]["dose_to_clear_noise_MGy"]),
            "ratio_to_lifetime_min": round(min(ratios), 2),
            "ratio_to_lifetime_max": round(max(ratios), 2),
            "single_crystal_viable": bool(max(ratios) < 1.0),
            "protein_growth_over_label": round(
                float(abs(g[mg]).max() / abs(lab[m]).max()), 1),
        })
    with open(out_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    return rows
