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
    figure_cryo,
    figure_dose_dependence,
    figure_qc,
    figure_signature,
)
from .pipeline import process_condition
from .quantify import band_series, integrate_band
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.toml",
                        help="experiment declaration (TOML)")
    parser.add_argument("--raw-dir", default=None,
                        help="override the raw .asc directory")
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--cross-validation", default=None,
                        help="cross-validation CSV for figure 2 panel (c); "
                             "omit to skip that panel's fitted-dose comparison")
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

    figure_signature(results, out / "figure_1_dtnb_signature.png")
    if args.cross_validation:
        cross = pd.read_csv(args.cross_validation)
        figure_dose_dependence(results, cross, out / "figure_2_dose_dependence.png")
    figure_cryo(registry, results, out / "figure_3_rt_vs_cryo.png")
    figure_qc(registry, results, dose_model, out / "figure_S1_quality_control.png")

    print(f"\nWrote figures and tables to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
