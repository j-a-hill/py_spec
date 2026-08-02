# crystalspec

Analysis of in-crystallo UV-visible absorption spectroscopy on R37S human
gamma-D-crystallin (HGD) microcrystals, with and without the thiol label
DTNB (Ellman's reagent), recorded during X-ray irradiation.

Beamtime `mx29074-47`, 22 September 2023. Andor SR303i spectrograph with a
DU970P EM-CCD, 0.1 s kinetic cycle, 12.4 keV X-rays.

## Quick start

```bash
pip install -e .
python -m crystalspec.run --config config/experiment.toml --out outputs/ \
    --cross-validation outputs/table_2_cross_validation.csv
```

That single command reads the raw `.asc` files and writes every figure and
table. No stage reads an intermediate CSV written by another, so the outputs
cannot drift out of step with each other.

Point `--raw-dir` at the beamline directory if it is not where
`config/experiment.toml` says.

## What the data are

Two geometries, and they are **not** interchangeable:

| | Room temperature | 100 K |
|---|---|---|
| Sample | thin film / chipless chip | single crystal |
| In the beam | several randomly oriented microcrystals **plus mother liquor** | one crystal |
| Replicate unit | **film position** (5 per condition) | crystal |
| Acquisition | continuous kinetic series, 1000 spectra at 0.1 s | single accumulations before and after each exposure |
| Dose axis | calibrated, continuous | exposure index only |

The room-temperature replicates are film positions, not crystals. The spread
between them reflects how much crystalline material and mother liquor
happened to sit in the beam, not the properties of any one crystal. Random
orientation is an advantage here: averaging over many orientations suppresses
the absorbance anisotropy that makes single-crystal in-crystallo
spectroscopy difficult.

Because the crystalline volume fraction in the probe beam is unknown and
varies between positions, **no amplitude in this analysis is converted to a
concentration or an absolute TNB occupancy.** Every reported quantity is a
difference or a ratio.

## Why the analysis is built this way

**The 412 nm TNB band is a shoulder, not a peak.** In solution TNB absorbs
at 412 nm. In these crystals that band never resolves; it sits on the flank
of a much larger feature near 330-345 nm. Fitting a free Gaussian there
returns a centre that wanders across the whole permitted range and an
amplitude that collapses to zero for roughly half the timepoints, so the
fit is describing noise. The signal is instead integrated over a fixed
395-430 nm window, which assumes nothing about band shape.

**Scattering dominates the raw signal.** A microcrystal scatters far more
light than it absorbs. At 700 nm, where nothing in this system absorbs, the
five positions of a single condition differ by a factor of two to four. Raw
absorbances are therefore never averaged across positions. Each position is
referenced to its own pre-irradiation dark state, and a power-law
background fitted on the non-absorbing 470-725 nm region is subtracted from
every spectrum.

**The detector baseline was reset during acquisition.** The operator reset
the baseline periodically to counter upward drift, which appears in the
data as a step across the whole spectrum. Isolated resets are stitched by
subtracting the wavelength-resolved offset across the step. Once resets
start arriving in bursts the trace is no longer a continuous measurement of
one sample state, so a burst ends the usable window instead. Both are
recorded per position in `table_S1_per_position_qc.csv` and drawn in
`figure_S1_quality_control.png`.

## Module layout

| Module | Responsibility |
|---|---|
| `io.py` | read Andor `.asc` (data block + instrument footer) |
| `registry.py` | which raw file is which measurement, from TOML |
| `dose.py` | frame index and attenuation to absorbed dose |
| `scatter.py` | power-law scattering background |
| `resets.py` | detect, stitch, and truncate at baseline resets |
| `pipeline.py` | per-condition processing, dark referencing, position stacking |
| `quantify.py` | band integration and bootstrap confidence intervals |
| `figures.py` | one function per figure |
| `style.py` | palette, sizing, overlap checking |
| `run.py` | regenerate everything |

The file-to-condition mapping lives in `config/experiment.toml`, not in code,
because the beamline naming is inconsistent (`hgd_R37S_test_100` has no
number, `hgd_R37S_tes5t_100` carries a typo) and an explicit table is
auditable.

## Outputs

| File | Contents |
|---|---|
| `figure_1_dtnb_signature.png` | DTNB-specific difference signature against matched apo controls |
| `figure_2_dose_dependence.png` | TNB signal against dose, per-position test statistic, characteristic dose |
| `figure_3_rt_vs_cryo.png` | room-temperature film ensemble against 100 K single crystals |
| `figure_S1_quality_control.png` | scatter correction, reset handling, usable window per position |
| `table_1_condition_summary.csv` | per condition: exposure, QC counts, endpoint band values |
| `table_2_cross_validation.csv` | band integration against SVD, both quantification routes |
| `table_3_label_specificity.csv` | DTNB against apo, exact permutation test per level |
| `table_S1_per_position_qc.csv` | per position: resets, stitching, truncation, usable window |

## Known limitations

- **Dose calibration is provisional.** The supplied RADDOSE-3D run used a
  50 x 50 um beam at 5e12 ph/s, while the chapter methods describe
  20 x 20 um at 8e12 ph/s for the I24 experiment. If the spectroscopy beam
  differed from the modelled one, every dose axis rescales by a constant
  factor. Comparisons *between* transmission levels are unaffected.
- **The internal-zero check is not flat.** After correction the 470-500 nm
  window should read zero at all doses. It does not: at the highest doses
  the residual reaches 12-29 % of the co-located TNB signal in the DTNB
  conditions. This is the precision floor on every integrated value.
- **The scatter exponent is weakly determined.** Held-out cross-validation
  prefers a power law over linear or constant backgrounds but barely
  distinguishes exponents. It is fixed at `n = 3`; spanning `n = 1..4`
  shifts the TNB integral by 11-60 % depending on condition (see
  `scatter.exponent_sensitivity`).
- **n = 5 positions per condition** is a weak base for the bootstrap. Some
  half-dose confidence intervals span more than an order of magnitude.
- **The third quantification route is inconclusive by construction, not by
  accident.** Reference-spectrum unmixing was run as an independent
  cross-check. It produced results, but they do not settle anything, for a
  reason worth stating: over the fit region the DTNB-depletion reference and
  the empirical protein-growth reference are anti-correlated at
  Pearson *r* = −0.97. The linear solver can trade one against the other
  almost for free, so band shifts and widths are not identifiable — a global
  fit drove them to the edges of their bounds for a 3 % improvement in
  residual. Band positions are therefore held at the literature values
  (TNB 412 nm, DTNB 325 nm) rather than reported as measured in-crystallo
  shifts.

  Two supporting problems compound it. The in-house DTNB reference film is
  real but far too weak to anchor the fit (SNR ≈ 31, yet only 2.6 % of the
  cryo single-crystal difference amplitude at the same nominal chemistry),
  and digitised literature spectra for DTNB and TNB were not retrievable
  open-access, so both reference bands are idealised Gaussians with assumed
  widths.

  What the route does show is informative. By BIC, adding the TNB and DTNB
  references earns their parameters only at 50 % and 100 % transmission; at
  5 % and 25 % the protein-plus-background null is preferred. Residuals
  concentrate at 286–300 nm, at the blue edge where the reference set has no
  component — a real gap in the description of the crystalline spectrum
  rather than a fitting failure.

  The two routes that do constrain the answer (band integration and SVD)
  agree on the sign and the presence of the effect, and disagree in detail
  on the characteristic dose above 50 % transmission.
