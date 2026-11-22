"""
style.py
========
The one place where the look of every figure, map and video is decided.
Scripts and notebooks import their colours from here and never write a hex code
of their own, so the whole project changes together.

    from ecuador_price import style
    style.apply()                       # fonts and axes, once per script/notebook
    ax.plot(x, y, color=style.FIRMS)

The four colours
----------------
Chosen to stay distinct for readers with red-green colour blindness (checked
with the palette validator: every pair is at least 12 apart under simulated
deuteranopia and protanopia, where 8 is the target), and each has at least 3:1
contrast on white.

    BLUE    #2e69b2      RED     #c92228      ORANGE  #d47d14      PURPLE  #743566

One colour, one meaning
-----------------------
A reader should learn the code once. Two families of meaning never share a
figure, so they can share colours:

    what is measured      FIRMS = SELECTION = blue     (the count side)
                          SALES = TRANSMISSION = red   (the money side)
                          TOTAL = ink
    which period          BASELINE = blue, SHOCK = red, AFTER = purple
    which city            Quito = purple, Guayaquil = orange, Cuenca = blue
    one thing against     HIGHLIGHT = red on CONTEXT grey
    the rest

Ordered groups (terciles, scales from province to parish) are NOT different
things, they are steps of one thing: they take one colour from light to dark
(``ramp``). The one exception is three groups drawn as curves that overlap in
the same panel, where steps of one colour cannot be told apart: those take
``GROUPS_3``, three colours from warm (lowest group) to cool (highest).

Maps and videos
---------------
One two-sided scale everywhere, the one of the Loja videos: dark red for a
fall, light grey at zero, dark blue for a rise. ``map_norm`` makes the scale
symmetric around zero and ``add_map_colorbar`` draws the same slim bar on the
right of every map.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb

# --------------------------------------------------------------------------- #
# The palette
# --------------------------------------------------------------------------- #
BLUE, RED, ORANGE, PURPLE = "#2e69b2", "#c92228", "#d47d14", "#743566"

INK = "#1a1a1a"          # text, axes, the "total" series, fitted lines
GREY = "#6b6b6b"         # secondary text, reference lines
CONTEXT = "#c9c9c9"      # dots and lines that are background to the story
NO_DATA = "#dbdbdb"      # map units with no data
BACKDROP = "#ededed"     # the rest of the country, behind the area being shown
EDGE = "#ffffff"         # outline between neighbouring map units and bars

# what is measured
FIRMS = SELECTION = BLUE
SALES = TRANSMISSION = RED
TOTAL = INK

# which period
BASELINE, SHOCK, AFTER = BLUE, RED, PURPLE
ERA = {"Pre-COVID": BASELINE, "COVID": SHOCK, "Post-COVID": AFTER}

# which city (the three metropolitan parishes)
CITY = {"Quito": PURPLE, "Guayaquil": ORANGE, "Cuenca": BLUE}

# one thing against the rest
HIGHLIGHT = RED

# the three ways of looking at a window
PATH = {"pop": BLUE, "inc": RED, "den": PURPLE}


def tint(color, amount):
    """``color`` mixed with white. amount = 0 returns the colour, 1 returns white.
    Fills under a line use tint(color, 0.8); never a second hand-picked hex."""
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def ramp(color, n):
    """``n`` steps of one colour from light to dark, for ORDERED groups
    (narrow < mid < broad; parish < canton < province). The last step is the
    colour itself."""
    lightest = 0.55
    return [tint(color, lightest * (n - 1 - i) / max(n - 1, 1)) for i in range(n)]


# the three scales of selection, from the finest (lightest) to the coarsest
SCALE = dict(zip(["parish", "canton", "province"], ramp(SELECTION, 3)))

# three ordered groups whose curves overlap in one panel: lowest, middle, highest
GROUPS_3 = [ORANGE, PURPLE, BLUE]


# --------------------------------------------------------------------------- #
# Maps and videos: one two-sided scale
# --------------------------------------------------------------------------- #
FALL, ZERO, RISE = "#822630", "#f2f2f2", "#265282"      # dark red, light grey, dark blue
DIVERGING = LinearSegmentedColormap.from_list("ecuador_diverging", [FALL, ZERO, RISE])
DIVERGING.set_bad(NO_DATA)


def map_norm(limit):
    """A colour scale symmetric around zero: -limit is full red, +limit full blue.
    Values beyond the limit take the end colour (the colour bar shows arrows)."""
    return Normalize(vmin=-limit, vmax=limit)


def add_map_colorbar(ax, limit, label="", ticks=5):
    """The colour bar of every map: slim, just right of ``ax``, 60 % of its
    height, arrows at both ends, the label on top. Returns the colour bar."""
    cax = ax.inset_axes([1.12, 0.20, 0.03, 0.60])      # clear of the map and of a place name at its edge
    bar = ax.figure.colorbar(ScalarMappable(norm=map_norm(limit), cmap=DIVERGING), cax=cax,
                             extend="both", extendfrac=0.04)
    step = 2 * limit / (ticks - 1)
    bar.set_ticks([-limit + i * step for i in range(ticks)])
    values = [-limit + i * step for i in range(ticks)]
    bar.ax.set_yticklabels(["0" if abs(v) < 1e-12 else f"{v:+.2g}".replace("-", "\u2212")
                            for v in values])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9, colors=INK)
    bar.ax.set_title(label, fontsize=9, color=INK, loc="left", pad=8)
    return bar


# --------------------------------------------------------------------------- #
# Fonts and axes
# --------------------------------------------------------------------------- #
RC = {
    "font.family": "STIXGeneral",          # the serif that matches the equations
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=[BLUE, RED, ORANGE, PURPLE]),
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}


def apply():
    """Set the fonts and axes of the project. Call once at the top of a script
    or notebook, before drawing."""
    plt.rcParams.update(RC)
