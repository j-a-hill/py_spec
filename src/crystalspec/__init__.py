"""crystalspec -- in-crystallo UV-visible absorption spectroscopy.

Analysis of X-ray-induced absorbance changes in R37S human gamma-D-crystallin
microcrystals, with and without the thiol label DTNB (Ellman's reagent),
measured on a fibre-coupled spectrograph during X-ray irradiation.

The package reads Andor ``.asc`` exports directly and reproduces every
figure and table from the raw beamline files in one command::

    python -m crystalspec.run --config config/experiment.toml --out outputs/

Design notes
------------
Two properties of these data drive the whole design.

*Scattering dominates.*  A microcrystal scatters far more than it absorbs,
and the scattered fraction varies two- to four-fold between crystal
positions within a single condition.  Absolute absorbances are therefore
never averaged across crystals; everything is done in difference space,
each crystal referenced to its own pre-irradiation dark state, with a
power-law scattering background removed per spectrum
(:mod:`crystalspec.scatter`).

*The TNB band is a shoulder, not a peak.*  In solution TNB absorbs at
412 nm, but in these crystals that band appears only as a shoulder on a
much larger feature near 330-345 nm.  Fitting a free Gaussian there
returns a centre that wanders across the whole permitted range and an
amplitude that collapses to zero -- it fits noise.  The TNB signal is
instead quantified three independent ways
(:mod:`crystalspec.quantify`) and the results cross-checked.
"""

from .dose import DoseModel, dose_axis
from .io import Acquisition, read_asc
from .pipeline import ConditionResult, CrystalTrace, process_condition
from .registry import Condition, Registry, load_registry
from .resets import ResetReport, correct_resets, detect_resets, stitch_resets
from .scatter import ScatterModel, exponent_sensitivity, fit_scatter, remove_scatter

__version__ = "0.1.0"

__all__ = [
    "Acquisition",
    "read_asc",
    "Condition",
    "Registry",
    "load_registry",
    "DoseModel",
    "dose_axis",
    "ScatterModel",
    "fit_scatter",
    "remove_scatter",
    "exponent_sensitivity",
    "ResetReport",
    "detect_resets",
    "stitch_resets",
    "correct_resets",
    "CrystalTrace",
    "ConditionResult",
    "process_condition",
]
