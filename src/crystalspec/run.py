"""Regenerate every figure and table from the raw ``.asc`` files.

    python -m crystalspec.run --config config/experiment.toml --out outputs/

Nothing is read from an intermediate CSV: each output is derived from the
beamline files in one pass, so the analysis cannot drift out of step with
itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .dose import DoseModel
from .figures import (
    figure_buffer_controls,
    figure_cryo,
    figure_diagnostic_wavelengths,
    figure_experiment_design,
    figure_original_idiom,
    figure_qc,
)
from .pipeline import process_condition
from .quantify import (
    band_series,
    integrate_band,
    table_diagnostic_wavelengths,
    table_dose_budget,
    table_exponential_fits,
)
from .viability import (
    figure_v1_signal_is_real,
    figure_v2_verdict,
    figure_v3_what_else_changes,
    table_viability,
)
from .registry import load_registry
from .style import apply_style

TNB_WINDOW = (395.0, 430.0)
UV_WINDOW = (300.0, 325.0)
ZERO_WINDOW = (470.0, 500.0)
PAIRS = [("DTNB_5", "apo_5"), ("DTNB_25", "apo_25"),
         ("DTNB_50", "apo_50"), ("DTNB_100", "apo_100")]


def per_position_auc(result, window, dose_max):
    """Dose-integrated band signal for each film position."""
    valid = result.valid() & (result.dose_MGy <= dose_max)
    per = integrate_band(result.wavelength_nm, result.delta_a[:, :, valid], window)
    dose = result.dose_MGy[valid]
    out = []
    for row in per:
        good = np.isfinite(row)
        out.append(np.trapezoid(row[good], dose[good]) if good.sum() > 1 else np.nan)
    return np.asarray(out)


def exact_permutation_test(x, y):
    """Two-sided exact permutation test on a difference of means.

    With five positions per group there are only 252 distinct splits, so
    the null distribution is enumerated exactly rather than sampled.  The
    smallest attainable two-sided p is 2/252 = 0.0079.
    """
    from itertools import combinations

    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    observed = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    n = x.size
    stats = [
        pool[list(idx)].mean() - pool[[i for i in range(pool.size) if i not in idx]].mean()
        for idx in combinations(range(pool.size), n)
    ]
    stats = np.asarray(stats)
    p = float((np.abs(stats) >= abs(observed) - 1e-12).mean())
    pooled_sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return observed, p, float(observed / pooled_sd) if pooled_sd > 0 else np.nan


def table_condition_summary(results, dose_model) -> pd.DataFrame:
    """One row per condition: exposure, QC and endpoint band values."""
    rows = []
    for key, res in results.items():
        tnb = band_series(res, TNB_WINDOW, "TNB")
        uv = band_series(res, UV_WINDOW, "UV")
        zero = band_series(res, ZERO_WINDOW, "zero")
        rows.append(dict(
            condition=key,
            soak=res.condition.soak,
            transmission_pct=res.condition.transmission_pct,
            n_positions=res.n_crystals,
            dose_rate_MGy_per_s=round(dose_model.rate_at(res.condition.transmission_pct), 4),
            max_analysed_dose_MGy=round(res.max_valid_dose_MGy(), 2),
            mean_onset_s=round(float(np.mean([t.onset_report.time_s for t in res.traces])), 1),
            n_optically_compromised=int(sum(t.optical_report.compromised for t in res.traces)),
            n_resets_detected=sum(t.reset_report.n_steps for t in res.traces),
            n_positions_truncated=int(sum(t.reset_report.truncated for t in res.traces)),
            tnb_dA_final=round(float(tnb.mean[-1]), 4),
            tnb_ci_lo=round(float(tnb.lo[-1]), 4),
            tnb_ci_hi=round(float(tnb.hi[-1]), 4),
            uv_growth_dA_final=round(float(uv.mean[-1]), 4),
            internal_zero_dA_final=round(float(zero.mean[-1]), 4),
            internal_zero_pct_of_tnb=round(
                abs(float(zero.mean[-1])) / max(abs(float(tnb.mean[-1])), 1e-9) * 100, 1),
        ))
    return pd.DataFrame(rows).sort_values(["soak", "transmission_pct"])


def table_label_specificity(results) -> pd.DataFrame:
    """DTNB versus matched apo at each transmission level."""
    rows = []
    for dk, ak in PAIRS:
        d, a = results[dk], results[ak]
        dose_max = min(d.max_valid_dose_MGy(), a.max_valid_dose_MGy())
        x = per_position_auc(d, TNB_WINDOW, dose_max)
        y = per_position_auc(a, TNB_WINDOW, dose_max)
        observed, p, cohens_d = exact_permutation_test(x, y)
        rows.append(dict(
            transmission_pct=d.condition.transmission_pct,
            common_dose_max_MGy=round(dose_max, 2),
            auc_DTNB_mean=round(float(np.nanmean(x)), 4),
            auc_apo_mean=round(float(np.nanmean(y)), 4),
            difference=round(float(observed), 4),
            cohens_d=round(cohens_d, 2),
            p_exact=round(p, 4),
            p_bonferroni=round(min(p * len(PAIRS), 1.0), 4),
            fully_separated=bool(np.nanmax(x) < np.nanmin(y)),
        ))
    return pd.DataFrame(rows)


def table_per_position_qc(results) -> pd.DataFrame:
    """One row per film position: resets, truncation, usable window."""
    rows = []
    for key, res in results.items():
        for i, tr in enumerate(res.traces, start=1):
            rep = tr.reset_report
            rows.append(dict(
                condition=key,
                soak=res.condition.soak,
                transmission_pct=res.condition.transmission_pct,
                position=i,
                file=tr.filename,
                onset_s=round(tr.onset_report.time_s, 1),
                onset_detected=tr.onset_report.detected,
                optically_compromised=tr.optical_report.compromised,
                max_red_excursion=round(float(tr.optical_report.max_excursion), 4),
                n_resets=rep.n_steps,
                n_stitched=int(rep.stitched_frames.size),
                truncated=rep.truncated,
                usable_s=round(rep.usable_frames * 0.1, 1),
                first_reset_s=(round(float(rep.step_times_s[0]), 1)
                               if rep.n_steps else np.nan),
                burst_start_s=(round(rep.burst_start_frame * 0.1, 1)
                               if rep.truncated else np.nan),
            ))
    return pd.DataFrame(rows)


def table_initial_slopes(results, fit_dose_max=2.0, n_boot=2000, seed=0) -> pd.DataFrame:
    """Initial slope of each band against dose, per condition.

    Fitted over a low-dose window common to every condition.  This replaces
    the whole-curve characteristic dose used earlier: the stretched
    exponential drove its shape parameter to the imposed bound in every
    DTNB condition, and the usable dose range now differs tenfold between
    conditions, so a slope over a shared window is both better defined and
    comparable across dose rates.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, res in results.items():
        dose_max = min(fit_dose_max, res.max_valid_dose_MGy())
        valid = res.valid() & (res.dose_MGy <= dose_max)
        dose = res.dose_MGy[valid]
        for window, label in ((TNB_WINDOW, "TNB_395_430"),
                              (UV_WINDOW, "UV_300_325")):
            per = integrate_band(res.wavelength_nm, res.delta_a[:, :, valid], window)
            slopes = []
            for row in per:
                good = np.isfinite(row)
                slopes.append(np.polyfit(dose[good], row[good], 1)[0]
                              if good.sum() >= 5 else np.nan)
            slopes = np.asarray(slopes)
            finite = slopes[np.isfinite(slopes)]
            if finite.size > 1:
                boots = [rng.choice(finite, finite.size, replace=True).mean()
                         for _ in range(n_boot)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
            else:
                lo = hi = np.nan
            rows.append(dict(
                condition=key, soak=res.condition.soak,
                transmission_pct=res.condition.transmission_pct,
                band=label, fit_dose_max_MGy=round(dose_max, 2),
                slope_per_MGy=round(float(np.nanmean(finite)), 5),
                ci95_lo=round(float(lo), 5), ci95_hi=round(float(hi), 5),
                n_positions=int(finite.size),
                excludes_zero=bool(np.isfinite(lo) and (lo > 0) == (hi > 0)),
            ))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.toml",
                        help="experiment declaration (TOML)")
    parser.add_argument("--raw-dir", default=None,
                        help="override the raw .asc directory")
    parser.add_argument("--out", default="outputs", help="output directory")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    registry = load_registry(args.config, raw_dir=args.raw_dir)
    problems = registry.check()
    if problems:
        print("Registry problems:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    dose_model = DoseModel.from_config(registry.dose)
    print(f"Dose model: {dose_model}")

    results = {}
    for key in registry.conditions:
        results[key] = process_condition(registry, key, dose_model)
        res = results[key]
        print(f"  {key:9s} {res.n_crystals} positions, "
              f"{int(res.valid().sum())} analysed frames, "
              f"to {res.max_valid_dose_MGy():.2f} MGy")

    summary = table_condition_summary(results, dose_model)
    summary.to_csv(out / "table_1_condition_summary.csv", index=False)
    specificity = table_label_specificity(results)
    specificity.to_csv(out / "table_3_label_specificity.csv", index=False)
    qc = table_per_position_qc(results)
    qc.to_csv(out / "table_S1_per_position_qc.csv", index=False)
    slopes = table_initial_slopes(results)
    slopes.to_csv(out / "table_4_initial_slopes.csv", index=False)

    # The diagnostic-wavelength route.  These three were computed outside the
    # package while the analysis was being settled; they are generated here so
    # that one command really does reproduce every table the chapter cites.
    diag = table_diagnostic_wavelengths(results, PAIRS)
    diag.to_csv(out / "table_8_diagnostic_wavelengths.csv", index=False)
    expo = table_exponential_fits(results, pairs=PAIRS)
    expo.to_csv(out / "table_9_exponential_fits_D90.csv", index=False)
    budget = table_dose_budget(results, PAIRS)
    budget.to_csv(out / "table_10_dose_budget.csv", index=False)

    # The minimal figure set.  Every figure here answers one question that the
    # chapter needs; the exploratory figures built during the analysis are not
    # regenerated because each is either superseded by one of these or is a
    # diagnostic whose conclusion is now stated in the methods text.
    figure_original_idiom(results, out / "figure_1_spectra_and_traces.png")
    figure_diagnostic_wavelengths(results, out / "figure_2_diagnostic_wavelengths.png")
    figure_experiment_design(results, out / "figure_3_experiment_design.png")
    figure_cryo(registry, results, out / "figure_4_rt_vs_cryo.png")

    # The viability set: three figures and one table that answer only
    # "could a joint diffraction/spectroscopy experiment work?".  They reuse
    # the dose budget computed above rather than re-deriving anything, so they
    # cannot drift from table 10.
    budget_rows = budget.to_dict("records")
    figure_v1_signal_is_real(results, PAIRS, registry,
                             out / "figure_V1_signal_is_real.png")
    figure_v2_verdict(results, PAIRS, budget_rows, out / "figure_V2_verdict.png")
    figure_v3_what_else_changes(results, PAIRS,
                                out / "figure_V3_what_else_changes.png")
    table_viability(results, PAIRS, budget_rows, out / "table_V1_viability.csv")

    figure_qc(registry, results, dose_model, out / "figure_S1_quality_control.png")
    buffers = registry.buffer_acquisitions()
    if buffers is not None:
        figure_buffer_controls(results, *buffers,
                               out / "figure_S2_buffer_controls.png")
    else:
        print("  note: buffer acquisitions not found; skipping figure S2")

    print(f"\nWrote figures and tables to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
