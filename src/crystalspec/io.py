"""Reading Andor Solis ``.asc`` exports.

An ``.asc`` file written by the Andor Solis software for a kinetic series
contains a numeric block followed by a free-text instrument footer::

    179.18542   0.102928   0.0162372   ...   (1 + N_spectra values)
    ...
    727.083     0.107099   0.108415    ...

    Date and Time:                Fri Sep 22 14:23:13 2023
    Acquisition Mode:             Kinetics
    Kinetic Cycle Time (secs):    0.1
    ...

The first column is wavelength in nm; the remaining columns are the
successive spectra of the kinetic series (a single column for an
``Accumulate`` acquisition).  Absorbance is exported directly when the
instrument ``Data Type`` is ``Optical Density``.

Nothing here mutates the data: :func:`read_asc` returns exactly what is on
disk, so every correction applied later is explicit and traceable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Acquisition", "read_asc"]

# A footer line is ``Key: value``.  Wavelength/absorbance rows never contain
# a colon, so this is an unambiguous discriminator.
_META_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")


@dataclass
class Acquisition:
    """One ``.asc`` file: a wavelength axis, a spectrum matrix and metadata.

    Attributes
    ----------
    wavelength_nm
        Wavelength axis, shape ``(n_wavelength,)``, ascending.
    absorbance
        Spectra, shape ``(n_wavelength, n_spectra)``.  For an ``Accumulate``
        acquisition ``n_spectra == 1``.
    meta
        Raw key/value pairs from the instrument footer, unparsed.
    path
        Source file.
    """

    wavelength_nm: np.ndarray
    absorbance: np.ndarray
    meta: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @property
    def n_spectra(self) -> int:
        return int(self.absorbance.shape[1])

    @property
    def mode(self) -> str:
        """``Kinetics`` or ``Accumulate``."""
        return self.meta.get("Acquisition Mode", "unknown")

    @property
    def cycle_time_s(self) -> float | None:
        """Seconds between successive spectra of a kinetic series."""
        v = self.meta.get("Kinetic Cycle Time (secs)")
        return float(v) if v is not None else None

    @property
    def exposure_s(self) -> float | None:
        v = self.meta.get("Exposure Time (secs)")
        return float(v) if v is not None else None

    @property
    def accumulations(self) -> int | None:
        v = self.meta.get("Number of Accumulations")
        return int(v) if v is not None else None

    @property
    def data_type(self) -> str:
        return self.meta.get("Data Type", "unknown")

    def time_s(self) -> np.ndarray:
        """Elapsed time of each spectrum, measured from the first.

        The Andor cycle time is the interval between the *starts* of
        successive exposures, so frame ``i`` begins at ``i * cycle_time``.
        """
        dt = self.cycle_time_s
        if dt is None:
            return np.zeros(self.n_spectra)
        return np.arange(self.n_spectra) * dt

    def __repr__(self) -> str:  # pragma: no cover - display only
        name = self.path.name if self.path else "<memory>"
        return (
            f"Acquisition({name!r}, {self.wavelength_nm.size} wavelengths x "
            f"{self.n_spectra} spectra, mode={self.mode})"
        )


def read_asc(path: str | Path) -> Acquisition:
    """Read one Andor ``.asc`` file.

    Parameters
    ----------
    path
        Path to the ``.asc`` file.

    Returns
    -------
    Acquisition
        Wavelength axis, absorbance matrix and instrument metadata.

    Raises
    ------
    ValueError
        If the file contains no parseable numeric rows, or if the numeric
        block is ragged in a way that cannot be resolved to a single width.

    Notes
    -----
    Rows are classified by attempting a float conversion of every
    whitespace-separated field.  This tolerates the ``\\r\\n`` line endings and
    the trailing blank lines that Solis writes, and does not require the
    caller to know the header or footer length in advance -- unlike a
    fixed ``skiprows`` count, which silently misreads a file whose footer
    length differs.
    """
    path = Path(path)
    rows: list[list[float]] = []
    meta: dict[str, str] = {}

    with path.open("r", encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            try:
                rows.append([float(x) for x in fields])
            except ValueError:
                m = _META_RE.match(line)
                if m:
                    key, value = m.group(1), m.group(2)
                    # Solis writes bare section headers such as "SR303i:".
                    if value:
                        meta[key] = value

    if not rows:
        raise ValueError(f"{path}: no numeric data rows found")

    # The numeric block is rectangular in every file we have seen; guard
    # against a truncated final row rather than letting NumPy build an
    # object array.
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        width = max(widths, key=lambda w: sum(len(r) == w for r in rows))
        dropped = sum(len(r) != width for r in rows)
        rows = [r for r in rows if len(r) == width]
        if dropped:
            import warnings

            warnings.warn(
                f"{path.name}: dropped {dropped} ragged row(s); "
                f"kept {len(rows)} rows of width {width}",
                stacklevel=2,
            )

    block = np.asarray(rows, dtype=float)
    wavelength = block[:, 0]
    absorbance = block[:, 1:]

    order = np.argsort(wavelength)
    if not np.array_equal(order, np.arange(wavelength.size)):
        wavelength = wavelength[order]
        absorbance = absorbance[order]

    return Acquisition(
        wavelength_nm=wavelength,
        absorbance=absorbance,
        meta=meta,
        path=path,
    )
