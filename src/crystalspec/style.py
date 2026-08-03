"""Shared figure style: palette, sizing, small helpers.

One place for every visual decision so that the thesis figures are
internally consistent and a change of journal format is a one-line edit.

Colour is bound to meaning and never reused for anything else:

``SOAK``
    DTNB-soaked versus unlabelled (apo) crystals.  A blue/orange pair,
    distinguishable under deuteranopia -- red/green would not be.
``TRANSMISSION``
    A single-hue ramp for the ordinal attenuator series, dark = more X-rays.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "SOAK",
    "SOAK_LABEL",
    "TRANSMISSION",
    "apply_style",
    "panel_label",
    "band_mask",
    "shade_band",
    "check_overlaps",
]

#: Soak state -> colour.
# DTNB and its TNB product are the yellow-orange chromophores, so the warm
# colour is bound to the DTNB arm and the cool colour to the unlabelled control.
# Changing this dict reassigns the colours everywhere; no figure hardcodes them.
SOAK = {"DTNB": "#c04a1e", "apo": "#1b3a6b"}
#: Soak state -> figure label.
SOAK_LABEL = {"DTNB": "DTNB-soaked", "apo": "apo (no DTNB)"}
#: Nominal X-ray transmission (%) -> colour, dark = higher dose rate.
TRANSMISSION = {
    5: "#bcd2e8",
    10: "#8fb4d9",
    25: "#5b8cc0",
    50: "#2f639f",
    100: "#12365f",
}

BASE, SMALL, TINY = 8, 7, 6


def apply_style() -> None:
    """Set rcParams for publication-grade output."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": BASE,
            "axes.titlesize": BASE,
            "axes.labelsize": BASE,
            "legend.fontsize": SMALL,
            "xtick.labelsize": TINY,
            "ytick.labelsize": TINY,
            "axes.titlelocation": "left",
            "axes.titlepad": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Bold panel letter outside the axes box."""
    ax.text(
        dx, dy, letter, transform=ax.transAxes,
        fontsize=BASE + 2, fontweight="bold", va="bottom", ha="left",
    )


def band_mask(wavelength_nm: np.ndarray, low: float, high: float) -> np.ndarray:
    """Boolean mask for a closed wavelength interval."""
    w = np.asarray(wavelength_nm, dtype=float)
    return (w >= low) & (w <= high)


def shade_band(ax, low: float, high: float, color: str = "0.90", label=None) -> None:
    """Shade a wavelength window behind the data."""
    ax.axvspan(low, high, color=color, lw=0, zorder=0, label=label)


def check_overlaps(fig) -> list[tuple[str, str]]:
    """Return pairs of visible text objects whose bounding boxes overlap.

    A tick label sitting on its own axis is not reported.  Use before
    saving: a non-empty result means labels collide and the layout needs
    fixing.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Matplotlib keeps detached offset-text placeholders on the figure that
    # report a bounding box but never draw (hiding them leaves the rendered
    # image bit-identical).  They have no parent axes; skipping them avoids
    # a permanent false positive.
    texts = [
        (t, t.get_window_extent(renderer))
        for t in fig.findobj(mpl.text.Text)
        if t.get_text().strip() and t.get_visible() and getattr(t, "axes", None) is not None
    ]
    own_ticks = {
        ax: set(ax.get_xticklabels() + ax.get_yticklabels()) for ax in fig.axes
    }
    out = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if not ba.overlaps(bb):
                continue
            ax_a = getattr(a, "axes", None)
            ax_b = getattr(b, "axes", None)
            if a in own_ticks.get(ax_b, ()) or b in own_ticks.get(ax_a, ()):
                continue
            if ax_a is ax_b and a in own_ticks.get(ax_a, ()) and b in own_ticks.get(ax_a, ()):
                continue
            out.append((a.get_text()[:40], b.get_text()[:40]))
    return out
