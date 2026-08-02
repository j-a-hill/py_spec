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

**Acquisition started before the shutter did.** Recording was begun by hand
a few seconds before the X-ray shutter was opened, and the delay was not
logged. Zeroing dose at frame 0 therefore attributes dose that was never
delivered, and because the error is a fixed *time* offset it scales with
dose rate: a few tenths of a MGy at 5 % transmission, several MGy at
100 %. The shutter is located per trace as the earliest persistent
departure in any spectral region, and dose accumulates from there. The
frames immediately preceding it form the dark reference.

The detector must scan all regions, not one band. An earlier version
triggered on the 300--325 nm damage band alone; that band responds slowly
at low dose rate, so at 5 % transmission it fired about 10 s after
irradiation had demonstrably begun, by which point the TNB window had
already fallen 0.02 absorbance units. Referencing to that late window
subtracted part of the signal and inverted the sign of the result.

**The sample was irradiated past destruction.** Exposures were pushed
until the EVAL film burned or crystals fractured. Beyond that point the
trace is still recorded but no longer measures absorbance. The 600--700 nm
window supplies the diagnostic: nothing in this system has a band there,
and the change seen at high dose rate is flat to 0.001 absorbance units
across 575--700 nm, so it reports bulk opacity rather than a chromophore.
Frames beyond a sustained crossing are excluded. This check runs *before*
scatter removal, because the power-law fit would otherwise absorb the very
offset that signals the damage.

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
| `onset.py` | locate the X-ray shutter within each series |
| `optical.py` | flag frames where the sample stopped being a valid absorbance sample |
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
| `figure_2_dose_dependence.png` | TNB signal against dose, per-position test statistic, initial slope |
| `figure_3_rt_vs_cryo.png` | room-temperature film ensemble against 100 K single crystals |
| `figure_S1_quality_control.png` | scatter correction, reset handling, usable window per position |
| `table_1_condition_summary.csv` | per condition: exposure, QC counts, endpoint band values |
| `table_2_cross_validation.csv` | band integration against SVD, both quantification routes |
| `table_4_initial_slopes.csv` | initial slope of each band against dose, with bootstrap intervals |
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
- **n = 5 positions per condition** is a weak base for the bootstrap.
- **The usable dose range differs tenfold between conditions** once
  optically compromised frames are excluded: 25 MGy at 25 % transmission
  against about 2 MGy at 50 % and 100 %. Whole-curve fits are correspondingly
  ill-conditioned, which is why the dose response is reported as an initial
  slope over a common low-dose window rather than as a fitted characteristic
  dose. The stretched-exponential fits used earlier drove their shape
  parameter to its bound in every DTNB condition and are not reported.
- **Absorbed dose is an estimate.** Beyond the beam-size question above, the
  shutter time is recovered from the data to a resolution of one 0.1 s frame,
  and later where the onset is gradual. The detected onset is best read as an
  upper bound on the true shutter time.
- **No signal below about 300 nm.** Absorbance *falls* from 0.28 at
  300--350 nm to 0.12 at 180--220 nm in both crystal and buffer, which cannot
  happen if light is reaching the detector, and frame-to-frame noise there is
  fifteen times worse than at 412 nm. The recorded pixels are a detector
  floor, not a measurement. Disulphide n->sigma* absorption near 250--260 nm
  is therefore *not* accessible in this dataset — not because the method
  cannot reach it, but because this optical configuration (150 l/mm grating
  blazed at 300 nm, centred at 454 nm, EVAL film and mother liquor in the
  path) did not deliver light there.
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

  What the route does show is informative. By BIC, a model carrying the
  DTNB-depletion reference beats the protein-plus-background null at 5 %
  (ΔBIC +10.1, DTNB-only the best of the four), 50 % (+51.7) and 100 %
  (+98.7) transmission; the null is preferred only at 25 % (−12.5). A TNB
  reference *on its own*, without DTNB, is worse than the null everywhere
  except 100 % transmission — so it is the DTNB-depletion component, not the
  TNB component, that carries the explanatory power in this decomposition.
  Residuals concentrate at 286–300 nm, at the blue edge where the reference
  set has no component — a real gap in the description of the crystalline
  spectrum rather than a fitting failure.

  The two routes that do constrain the answer (band integration and SVD)
  agree on the sign and the presence of the effect, and disagree in detail
  on the characteristic dose above 50 % transmission.
