"""Experiment registry: which raw file is which measurement.

The beamline filenames encode the sample and the X-ray attenuation but not
in a single consistent scheme -- DTNB-soaked crystals are numbered
``hgd_R37S_DTNB_<n>_<transmission>_transmission.asc`` while the matched
unlabelled controls are ``hgd_R37S_test<n>_<transmission>_transmission.asc``,
with two files departing from even that (``hgd_R37S_test_100`` has no
number and ``hgd_R37S_tes5t_100`` carries a typo).  Rather than parse the
names, the mapping is declared explicitly in a TOML config so that the
association between a file and its experimental condition is auditable and
can be corrected without touching code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .io import Acquisition, read_asc

__all__ = ["Condition", "Registry", "load_registry"]


@dataclass(frozen=True)
class Condition:
    """One experimental condition: a soak state at one X-ray transmission.

    Attributes
    ----------
    key
        Identifier such as ``DTNB_5`` or ``apo_100``.
    label
        Human-readable name for figures, e.g. ``"DTNB, 5% T"``.
    soak
        ``"DTNB"`` or ``"apo"``.
    transmission_pct
        Nominal X-ray transmission through the attenuator, per cent.
    files
        Raw ``.asc`` filenames, one per crystal position.
    """

    key: str
    label: str
    soak: str
    transmission_pct: float
    files: tuple[str, ...]

    @property
    def n_crystals(self) -> int:
        return len(self.files)


@dataclass
class Registry:
    """All declared conditions plus the reference and cryo measurements."""

    raw_dir: Path
    conditions: dict[str, Condition]
    references: dict[str, str]
    cryo: dict[str, list[str]]
    dose: dict[str, float]

    def path(self, filename: str) -> Path:
        return self.raw_dir / filename

    def load(self, filename: str) -> Acquisition:
        return read_asc(self.path(filename))

    def load_condition(self, key: str) -> list[Acquisition]:
        """Load every crystal of one condition, in declared order."""
        return [self.load(f) for f in self.conditions[key].files]

    def check(self) -> list[str]:
        """Return a list of problems: missing files, unreadable acquisitions.

        An empty list means every declared file is present and parses.
        """
        problems: list[str] = []
        declared: list[str] = []
        for cond in self.conditions.values():
            declared.extend(cond.files)
        declared.extend(self.references.values())
        for group in self.cryo.values():
            declared.extend(group)

        for filename in declared:
            p = self.path(filename)
            if not p.exists():
                problems.append(f"missing: {filename}")
                continue
            try:
                read_asc(p)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                problems.append(f"unreadable: {filename}: {exc}")
        return problems

    def by_soak(self, soak: str) -> list[Condition]:
        """Conditions with the given soak state, ordered by transmission."""
        got = [c for c in self.conditions.values() if c.soak == soak]
        return sorted(got, key=lambda c: c.transmission_pct)

    def buffer_acquisitions(self, static: str = "buffer_RT.asc",
                            irradiated: str = "buffer_RT_xrays.asc"):
        """Return ``(static, irradiated)`` mother-liquor acquisitions, or ``None``.

        These are the blank/reference acquisitions taken at the beamline rather
        than per-condition measurements, so they are not in the condition table.
        Returns ``None`` when either file is absent so a caller can skip the
        control figure rather than fail.
        """
        paths = [self.raw_dir / static, self.raw_dir / irradiated]
        if not all(p.exists() for p in paths):
            return None
        return tuple(read_asc(p) for p in paths)

    def matched_pairs(self) -> list[tuple[Condition, Condition]]:
        """(DTNB, apo) condition pairs sharing a transmission level."""
        apo = {c.transmission_pct: c for c in self.by_soak("apo")}
        return [
            (d, apo[d.transmission_pct])
            for d in self.by_soak("DTNB")
            if d.transmission_pct in apo
        ]


def load_registry(config_path: str | Path, raw_dir: str | Path | None = None) -> Registry:
    """Build a :class:`Registry` from a TOML config file.

    Parameters
    ----------
    config_path
        Path to ``experiment.toml``.
    raw_dir
        Directory holding the ``.asc`` files.  Overrides the ``raw_dir`` key
        in the config, which is interpreted relative to the config file.

    Returns
    -------
    Registry
    """
    config_path = Path(config_path)
    with config_path.open("rb") as fh:
        cfg = tomllib.load(fh)

    if raw_dir is not None:
        root = Path(raw_dir)
    else:
        root = (config_path.parent / cfg["raw_dir"]).resolve()

    conditions = {}
    for key, spec in cfg["conditions"].items():
        conditions[key] = Condition(
            key=key,
            label=spec["label"],
            soak=spec["soak"],
            transmission_pct=float(spec["transmission_pct"]),
            files=tuple(spec["files"]),
        )

    return Registry(
        raw_dir=root,
        conditions=conditions,
        references=dict(cfg.get("references", {})),
        cryo={k: list(v) for k, v in cfg.get("cryo", {}).items()},
        dose=dict(cfg.get("dose", {})),
    )
