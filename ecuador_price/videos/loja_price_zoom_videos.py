"""Six split-screen zoom videos of the Price terms behind Loja's flat 2023-2024 average.

The companion to `loja_zoom_videos.py`. That one maps what an average sees --
each unit's own Dln(real sales). This one maps what the average cannot see: the
two terms the Price equation splits that change into, from STEP F of
`masked_growth_multiscale_maps.ipynb`.

    Dz-bar  =  S (selection)  +  T (transmission)
    sel     =  p (w/W-bar - 1) (z0 - z0-bar)      trans = q (z1 - z0)

Every frame carries both terms side by side over the identical camera move, so
the two panels can only ever differ in what the decomposition says. Nine videos,
cut out of one continuous camera move: each starts on the exact frame the
previous one ended on, so they play back-to-back or stand on their own.

  1  ecuador_to_loja        Ecuador at province scale -> zoom into Loja.
  2  loja_into_cantons      Loja dissolves into its 16 cantons.
  3,5,7,9  one per regime   Zoom from canton scale into the canton holding the
                            parish that best represents that regime, which then
                            dissolves into parishes -- and everything except the
                            parish under analysis fades to the backdrop grey.
  4,6,8  the way back       The intermediate chapters, one between each pair of
                            regimes: the parishes re-assemble into their canton,
                            the greyed-out map comes back, the camera pulls out to
                            Loja, and the outline hands off to the next regime's
                            canton. Expansion -> consolidation -> dilution ->
                            abandonment, so the four regimes are one journey.

The four regimes are the sign pairs of (sel, trans), and the exemplars are
picked from the notebook's own ledger -- see CASES below for who and why.

COLOUR. Red = a NEGATIVE term (this unit pulls the average down), blue = a
POSITIVE one, and the midpoint is a true neutral so that zero looks like zero.
Read the colour as a sign, not as good news -- an S- parish is *gaining* firm
share, and it is red.

TWO CAPS, one per scope. A Price term is a contribution to the mean-trait change
of whatever scope it was computed in: the province maps decompose ECUADOR's
change, the canton and parish maps decompose LOJA's own. Those are different
decompositions, and they are not the same size --

    max|sel|    province 0.0020   canton 0.0128   parish 0.0152
    max|trans|  province 0.0053   canton 0.0122   parish 0.0126

-- so one cap cannot serve both scopes. At the Loja cap the whole of Ecuador
lands inside 13% of the half-scale and the country renders as one flat colour.
Each scope gets the cap that fits it; within a scope BOTH terms share a cap,
which is what makes the two panels comparable to each other. Set the two caps
equal to go back to a single scale.

The frame carries the two titles, the two colour bars, and place names. Nothing
else: a zoom does not need a caption saying it is a zoom, and the numbers belong
in the talk. `main()` prints them to stdout instead.

Data note: 2024 is missing November, so every comparison uses the 11 months the
two years share.
"""

import sys
import time

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba

from ecuador_price import decomposition as dec
from ecuador_price import paths
from ecuador_price import style

# --------------------------------------------------------------------------- #
# Parameters -- the whole story is retargetable from here.
# --------------------------------------------------------------------------- #
Y0, Y1 = 2023, 2024               # the window; the decomposition is Y0 -> Y1
SHARED_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]   # months present in both years

PROV_CODE = "11"                  # Loja -- the province with the flat average

# The regime exemplars, ranked out of the notebook's own ledger. Each entry is
# (parish, its canton, why this one). Swapping a pick is a one-line edit.
CASES = {
    "expansion":     ("111350", "1113",   # Zapotillo -- S -0.0091, T +0.0041
                      "sales +38%, firms 37->45: both terms large, the textbook "
                      "boomtown. (Alternative: Celica, bigger S but T ~ 0.)"),
    "consolidation": ("111450", "1114",   # Pindal -- S +0.0016, T +0.0126
                      "sales x6.7 on the same 24 firms: the regime's signature is "
                      "T alone. (Alternative: Saraguro, S and T evenly matched.)"),
    "dilution":      ("111159", "1111",   # Urdaneta -- S -0.0048, T -0.0018
                      "firms +35% while sales -38%, the sharpest contrast in the "
                      "quadrant. (Alternative: Alamor, same mass on a bigger dec.)"),
    "abandonment":   ("110655", "1106",   # El Ingenio -- S +0.0053, T -0.0016
                      "firms -50% and sales -88%: decay on both axes at once. "
                      "(Alternative: Olmedo, biggest entry but its firms stayed.)"),
}

# The two panels: (column in the price table, title, colour-bar label).
PANELS = [("sel_c",   "Selection",    "sel (log points)"),
          ("trans_c", "Transmission", "trans (log points)")]

# Symmetric colour caps in log points of Δz̄, one per scope -- see the docstring.
CAP_NATIONAL = 0.006              # province rows: max|term| there is 0.0053
CAP_LOJA     = 0.016              # canton/parish rows: max|term| there is 0.0152
# The project's two-sided map scale (ecuador_price/style.py): dark red = fall,
# light grey = zero, dark blue = rise. The midpoint is NEUTRAL on purpose: a
# tinted midpoint renders a zero term as a positive one.
CMAP = style.DIVERGING
NO_DATA = style.NO_DATA           # units the decomposition could not use
BACKDROP = style.BACKDROP         # the rest of the country, for orientation
MUTED = BACKDROP                  # and what everything we are NOT analysing fades to
FPS = 30

# 2560x1080: two map panels side by side, each with its own colour bar. The
# widths are set by the tick labels, not the maps: "-0.016  (-1.59%)" is ~184px,
# and two of those have to fit between the panels and inside the right edge.
FIG_W, FIG_H, DPI = 16.0, 6.75, 160
AX_RECTS   = [[0.006, 0.030, 0.390, 0.845], [0.500, 0.030, 0.390, 0.845]]
CBAR_RECTS = [[0.406, 0.190, 0.009, 0.560], [0.900, 0.190, 0.009, 0.560]]
CRF_QUALITY = 9                   # imageio 1-10; 9 -> h264 crf 5, visually lossless
MACRO_BLOCK = 8                   # 1080 is not a multiple of 16: the default pads to 1088

OUT_DIR = paths.STORY / "loja_price_zoom"
SAVE_FRAMES = False               # also dump every frame as a PNG
ONLY = ()                         # videos to render (1-9); empty = all. Command line
                                  # wins: `loja_price_zoom_videos.py 4 6 8` redoes just
                                  # the three intermediate chapters.



# --------------------------------------------------------------------------- #
# Data: the Price terms per unit, at any scale, optionally inside one province
# --------------------------------------------------------------------------- #
def price_table(panel, width, within=None):
    """STEP D of the notebook: one row per unit (= first `width` digits of the
    parish code) with its own growth and its selection and transmission terms in
    the mean-trait change of the scope. `within` restricts the decomposition to
    one province, so the terms then decompose THAT province."""
    d = panel[panel["year"].isin([Y0, Y1])]
    if 2024 in (Y0, Y1):
        d = d[d["month"].isin(SHARED_MONTHS)]
    d = d.assign(unit=d["cod_parr"].str[:width])
    if within is not None:
        d = d[d["unit"].str.startswith(within)]
    sales = d.groupby(["unit", "year"])["real_sales"].sum().unstack()
    firms = (d.groupby(["unit", "year", "month"])["total_firms"].sum()   # sections summed
               .groupby(["unit", "year"]).mean().unstack())              # -> mean monthly stock
    ok = (sales > dec.SALES_FLOOR).all(axis=1) & (firms > 0).all(axis=1)
    surv = sales.index[ok]
    n0, n1 = firms.loc[surv, Y0], firms.loc[surv, Y1]
    z0, z1 = np.log(sales.loc[surv, Y0]), np.log(sales.loc[surv, Y1])
    p, q = n0 / n0.sum(), n1 / n1.sum()
    w = n1 / n0
    W, Z0 = (p * w).sum(), (p * z0).sum()
    t = pd.DataFrame({"growth":  z1 - z0,
                      "sel_c":   p * (w / W - 1) * (z0 - Z0),
                      "trans_c": q * (z1 - z0),
                      "n0": n0, "n1": n1, "s0": sales.loc[surv, Y0]})
    dzbar = (q * z1).sum() - Z0
    assert abs(t.sel_c.sum() + t.trans_c.sum() - dzbar) < 1e-9, "Price identity broken"
    return t


def pct(logpts):
    """Log points as the percentage a reader expects."""
    return np.expm1(logpts) * 100


# --------------------------------------------------------------------------- #
# Camera: a window in map units that covers a set of bounds at a panel's aspect
# --------------------------------------------------------------------------- #
AX_ASPECT = (FIG_W * AX_RECTS[0][2]) / (FIG_H * AX_RECTS[0][3])   # width / height


def camera(bounds, pad, bias=0.45):
    """(cx, cy, half_height) covering `bounds` with a `pad` fractional margin.

    Most of these shapes are taller than a panel, so they can only fill it
    vertically. `bias` slides the leftover width to the left, parking the map
    beside its colour bar instead of stranding it in the middle of the panel."""
    minx, miny, maxx, maxy = bounds
    half = max((maxy - miny) / 2, (maxx - minx) / 2 / AX_ASPECT) * (1 + pad)
    slack = half * AX_ASPECT - (maxx - minx) / 2 * (1 + pad)
    return np.array([(minx + maxx) / 2 - slack * bias, (miny + maxy) / 2, half])


def smoothstep(t):
    """Ease-in/ease-out on t in [0, 1]."""
    return t * t * (3 - 2 * t)


def interp_cam(c0, c1, t):
    """Pan linearly but zoom geometrically -- a linear `half` would make a 20x
    zoom look like it accelerates off a cliff at the end."""
    cx, cy = c0[:2] + (c1[:2] - c0[:2]) * t
    half = np.exp(np.log(c0[2]) + (np.log(c1[2]) - np.log(c0[2])) * t)
    return cx, cy, half


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
NO_DATA_RGBA, MUTED_RGBA = to_rgba(NO_DATA), to_rgba(MUTED)


def scale_for(cap):
    """The colour scale for one scope: its mappable and its five tick values."""
    sm = ScalarMappable(norm=Normalize(vmin=-cap, vmax=cap), cmap=CMAP)
    return sm, [-cap, -cap / 2, 0.0, cap / 2, cap]


def colors_for(geo, key, table, term, sm, keep=None, mute=0.0):
    """Per-polygon fill: the unit's Price term on this scope's scale, grey if the
    decomposition could not use it. Every unit other than `keep` is faded toward
    the backdrop grey by `mute` (0 = untouched, 1 = fully greyed), which is how
    the map empties out around the one unit under analysis. All RGBA -- geopandas
    needs one dtype."""
    out = []
    for code, v in zip(geo[key], geo[key].map(table[term])):
        c = NO_DATA_RGBA if pd.isna(v) else sm.to_rgba(v)
        if mute > 0 and code != keep:
            c = tuple(ci * (1 - mute) + mi * mute for ci, mi in zip(c, MUTED_RGBA))
        out.append(c)
    return out


def draw_layer(ax, geo, key, table, term, sm, alpha, lw=0.5, keep=None, mute=0.0):
    if alpha <= 0.002 or geo.empty:
        return
    geo.plot(color=colors_for(geo, key, table, term, sm, keep, mute), ax=ax,
             alpha=alpha, edgecolor="white", linewidth=lw)


def wrap(name, width=13):
    """Break a long name across lines so its label stays inside its own polygon."""
    lines, line = [], ""
    for word in name.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(lines)


def draw_labels(ax, geo, key, names, alpha, view, fontsize=9.0, codes=None, big=None):
    """Name every unit whose label point is inside the current view; `big` is the
    one unit the frame is about, which gets a larger label."""
    if alpha <= 0.02:
        return
    (x0, x1), (y0, y1) = view
    for _, r in geo.iterrows():
        if codes is not None and r[key] not in codes:
            continue
        pt = r.geometry.representative_point()
        if not (x0 < pt.x < x1 and y0 < pt.y < y1):
            continue
        ax.annotate(wrap(names.get(r[key], r[key])), (pt.x, pt.y),
                    ha="center", va="center", fontweight="bold",
                    fontsize=fontsize + (2.0 if r[key] == big else 0.0),
                    color=style.INK, alpha=min(alpha, 1.0),
                    path_effects=HALO, zorder=6)


def outline(ax, geo, key, code, alpha, lw=2.2):
    if alpha <= 0.02:
        return
    sel = geo[geo[key] == code]
    if len(sel):
        sel.boundary.plot(ax=ax, color="black", linewidth=lw,
                          alpha=min(alpha, 1.0), zorder=5)


# --------------------------------------------------------------------------- #
# One panel, then one frame
# --------------------------------------------------------------------------- #
def draw_panel(fig, slot, G, T, term, title, sm, ticks, cam,
               w_prov, w_cant, w_parr, w_mute, focus_cant, solo_parr, marks):
    """One of the two side-by-side maps. `slot` indexes AX_RECTS/CBAR_RECTS."""
    cx, cy, half = cam
    view = ((cx - half * AX_ASPECT, cx + half * AX_ASPECT), (cy - half, cy + half))
    ax = fig.add_axes(AX_RECTS[slot])

    # the rest of the country, always, so you never lose your bearings
    G["prov"].plot(color=BACKDROP, ax=ax, edgecolor="white", linewidth=0.4, zorder=0)

    draw_layer(ax, G["prov"], "DPA_PROVIN", T["prov"], term, sm, w_prov, lw=0.4)
    draw_layer(ax, G["cant"], "DPA_CANTON", T["cant"], term, sm, w_cant, lw=0.6,
               keep=focus_cant, mute=w_mute)
    if w_parr > 0 and focus_cant is not None:
        draw_layer(ax, G["parr"][G["parr"]["DPA_CANTON"] == focus_cant],
                   "DPA_PARROQ", T["parr"], term, sm, w_parr, lw=0.6,
                   keep=solo_parr, mute=w_mute)

    for level, code, a in marks:
        key = {"prov": "DPA_PROVIN", "cant": "DPA_CANTON", "parr": "DPA_PARROQ"}[level]
        outline(ax, G[level], key, code, a, lw=2.4 if level == "parr" else 2.0)

    # provinces go unnamed -- the outline and the zoom already say which one it is.
    # Inside a case canton only the analysed parish is named; the greyed ones are
    # there to be seen, not read.
    draw_labels(ax, G["cant"], "DPA_CANTON", NAMES["cant"], w_cant * (1 - w_parr), view)
    if focus_cant is not None:
        draw_labels(ax, G["parr"][G["parr"]["DPA_CANTON"] == focus_cant],
                    "DPA_PARROQ", NAMES["parr"], w_parr, view, fontsize=8.5,
                    codes=None if solo_parr is None else {solo_parr}, big=solo_parr)

    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(AX_RECTS[slot][0], 0.940, title, fontsize=16,
             fontweight="bold", color="#111")

    bx, by, _, bh = CBAR_RECTS[slot]
    cax = fig.add_axes(CBAR_RECTS[slot])
    cb = fig.colorbar(sm, cax=cax, ticks=ticks, extend="both")
    cb.ax.set_yticklabels([f"{v:+.3f}  ({pct(v):+.2f}%)" for v in ticks], fontsize=9)
    cb.outline.set_visible(False)
    cax.tick_params(length=2.5)
    fig.text(bx, by + bh + 0.035, LABELS[term], fontsize=10.5, color="0.25")


def render(G, T, sms, ticks, cam, weights, focus_cant, solo_parr, marks):
    """Both panels over the identical camera, so only the decomposition differs."""
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    for slot, (term, title, _) in enumerate(PANELS):
        draw_panel(fig, slot, G, T, term, f"{title} ({Y0} - {Y1})",
                   sms[term], ticks, cam, *weights, focus_cant, solo_parr, marks)
    canvas = fig.canvas
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global NAMES, LABELS

    LABELS = {term: label for term, _, label in PANELS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = dec.load_real_panel()
    panel["month"] = panel["mes"].astype(int)

    # --- the three tables the maps are coloured by --------------------------
    # province rows decompose ECUADOR's mean-trait change; canton and parish
    # rows decompose LOJA's own. See the module docstring.
    T = {"prov": price_table(panel, 2),
         "cant": price_table(panel, 4, within=PROV_CODE),
         "parr": price_table(panel, 6, within=PROV_CODE)}

    # --- geometry -----------------------------------------------------------
    prov = gpd.read_file(paths.SHAPEFILES["province"])
    cant = gpd.read_file(paths.SHAPEFILES["canton"])
    parr = gpd.read_file(paths.SHAPEFILES["parish"])
    G = {"prov": prov,
         "cant": cant[cant["DPA_PROVIN"] == PROV_CODE].copy(),
         "parr": parr[parr["DPA_PROVIN"] == PROV_CODE].copy()}
    NAMES = {"cant": {r.DPA_CANTON: r.DPA_DESCAN.title() for _, r in G["cant"].iterrows()},
             "parr": {r.DPA_PARROQ: r.DPA_DESPAR.title().split(" (")[0]
                      for _, r in G["parr"].iterrows()}}

    # --- the ledger, printed rather than drawn ------------------------------
    for lvl, lab in [("prov", f"Ecuador @ province ({len(T['prov'])} units)"),
                     ("cant", f"Loja @ canton ({len(T['cant'])} units)"),
                     ("parr", f"Loja @ parish ({len(T['parr'])} units)")]:
        t = T[lvl]
        print(f"{lab:32s} S {t.sel_c.sum():+.4f}  T {t.trans_c.sum():+.4f}  "
              f"Δz̄ {t.sel_c.sum() + t.trans_c.sum():+.4f}   "
              f"max|sel| {t.sel_c.abs().max():.4f}  max|trans| {t.trans_c.abs().max():.4f}")
    for lvl, cap, lab in [("prov", CAP_NATIONAL, "national scope, cap"),
                          ("cant", CAP_LOJA, "Loja scope,     cap")]:
        t = T[lvl]
        used = max(t.sel_c.abs().max(), t.trans_c.abs().max()) / cap
        print(f"  {lab} ±{cap}: widest term uses {used:.0%} of the half-scale")
    print(f"\nLoja's own row in the national decomposition: "
          f"sel {T['prov'].loc[PROV_CODE, 'sel_c']:+.4f}, "
          f"trans {T['prov'].loc[PROV_CODE, 'trans_c']:+.4f}\n")
    print("the four regime exemplars:")
    for regime, (pc, cc, why) in CASES.items():
        r = T["parr"].loc[pc]
        print(f"  {regime:14s} {NAMES['parr'][pc]:<12s} ({NAMES['cant'][cc]}, "
              f"{int((G['parr'].DPA_CANTON == cc).sum())} parishes)   "
              f"sel {r.sel_c:+.4f}  trans {r.trans_c:+.4f}   "
              f"sales {pct(r.growth):+.0f}%  firms {r.n0:.0f}→{r.n1:.0f}")
        print(f"                 {why}")

    # --- cameras ------------------------------------------------------------
    CAM_EC   = camera(prov.total_bounds, pad=0.03)
    CAM_LOJA = camera(prov[prov.DPA_PROVIN == PROV_CODE].total_bounds, pad=0.10)
    CAM_CASE = {regime: camera(G["cant"][G["cant"].DPA_CANTON == cc].total_bounds, pad=0.30)
                for regime, (_, cc, _) in CASES.items()}

    s = lambda sec: max(int(round(sec * FPS)), 1)
    MARK_LOJA = ("prov", PROV_CODE, 1.0)

    # --- the chapters -------------------------------------------------------
    # a stage is (seconds, cam0, cam1, weights0, weights1, focus canton,
    #             solo parish, marks); weights = (provinces, Loja's cantons,
    #             the focus canton's parishes, how far everything we are NOT
    #             analysing has faded to grey). A mark's alpha may be a
    #             (from, to) pair, so outlines fade in and out too.
    # No captions: a hold is a beat, a zoom explains itself.
    P = (1, 0, 0, 0)      # Ecuador's provinces
    C = (0, 1, 0, 0)      # Loja's cantons, all of them in colour
    R = (0, 1, 1, 1)      # the focus canton opened, everything else greyed out

    ecuador_to_loja = [
        (1.6, CAM_EC, CAM_EC, P, P, None, None, [MARK_LOJA]),
        (3.6, CAM_EC, CAM_LOJA, P, P, None, None, [MARK_LOJA]),
        (1.6, CAM_LOJA, CAM_LOJA, P, P, None, None, [MARK_LOJA]),
    ]
    loja_into_cantons = [
        (1.2, CAM_LOJA, CAM_LOJA, P, P, None, None, [MARK_LOJA]),
        (2.4, CAM_LOJA, CAM_LOJA, P, C, None, None, [MARK_LOJA]),
        (3.2, CAM_LOJA, CAM_LOJA, C, C, None, None, [MARK_LOJA]),
    ]

    def case_chapter(regime, fade_in=True):
        """Canton scale -> into the regime's canton -> its parishes, with only
        the analysed one left in colour.

        `fade_in` draws the canton's outline in from nothing. Only the first
        regime needs it: after that the preceding back_chapter has already handed
        the outline over, and re-fading it would blink between the two videos."""
        pc, cc, _ = CASES[regime]
        cam = CAM_CASE[regime]
        return [
            (1.4, CAM_LOJA, CAM_LOJA, C, C, None, None,
             [MARK_LOJA, ("cant", cc, (0.0, 1.0) if fade_in else 1.0)]),
            (3.4, CAM_LOJA, cam, C, C, None, None,
             [("prov", PROV_CODE, (1.0, 0.0)), ("cant", cc, 1.0)]),
            (1.0, cam, cam, C, C, None, None, [("cant", cc, 1.0)]),
            (2.2, cam, cam, C, R, cc, pc,
             [("cant", cc, (1.0, 0.35)), ("parr", pc, (0.0, 1.0))]),
            (2.8, cam, cam, R, R, cc, pc,
             [("cant", cc, 0.35), ("parr", pc, 1.0)]),
        ]

    def back_chapter(regime, nxt):
        """The way back out, from one regime's parish to the next regime's canton.

        Runs case_chapter's last three stages in reverse -- hold on the parish,
        let the parishes re-assemble and the greyed-out map come back (R -> C
        returns the mute to 0), pull the camera out to Loja -- and then swaps the
        outline from this canton to the next one, which is the frame the next
        video opens on."""
        pc, cc, _ = CASES[regime]
        cc_next = CASES[nxt][1]
        cam = CAM_CASE[regime]
        return [
            (1.0, cam, cam, R, R, cc, pc,
             [("cant", cc, 0.35), ("parr", pc, 1.0)]),
            (1.6, cam, cam, R, C, cc, pc,
             [("cant", cc, (0.35, 1.0)), ("parr", pc, (1.0, 0.0))]),
            (3.4, cam, CAM_LOJA, C, C, None, None,
             [("prov", PROV_CODE, (0.0, 1.0)), ("cant", cc, 1.0)]),
            (1.8, CAM_LOJA, CAM_LOJA, C, C, None, None,
             [MARK_LOJA, ("cant", cc, (1.0, 0.0)), ("cant", cc_next, (0.0, 1.0))]),
        ]

    # --- the nine videos ----------------------------------------------------
    # Chapter 1 never leaves province scale, so it takes the national cap.
    # Everything from chapter 2 on is about Loja's own decomposition.
    # The regimes alternate with the way back, in CASES order, so the whole set
    # is one journey: expansion -> back -> consolidation -> back -> ... Reorder
    # or rename CASES and the sequence follows.
    slug = lambda name: name.lower().replace(" ", "_")
    regimes = list(CASES)
    videos = [("ecuador_to_loja", ecuador_to_loja, CAP_NATIONAL),
              ("loja_into_cantons", loja_into_cantons, CAP_LOJA)]
    for i, regime in enumerate(regimes):
        videos.append((f"{regime}_{slug(NAMES['parr'][CASES[regime][0]])}",
                       case_chapter(regime, fade_in=(i == 0)), CAP_LOJA))
        if i + 1 < len(regimes):
            nxt = regimes[i + 1]
            videos.append((f"back_{regime}_to_{nxt}",
                           back_chapter(regime, nxt), CAP_LOJA))

    frames_dir = OUT_DIR / "frames"
    if SAVE_FRAMES:
        frames_dir.mkdir(exist_ok=True)
    total_frames = sum(s(st[0]) for _, stages, _ in videos for st in stages)
    print(f"\n{len(videos)} split-screen videos, {total_frames} frames "
          f"({total_frames / FPS:.0f}s at {FPS} fps, {FIG_W * DPI:.0f}x{FIG_H * DPI:.0f})")
    only = {int(a) for a in sys.argv[1:]} or set(ONLY)
    if only:
        print(f"rendering only video(s) {sorted(only)}; the rest are left alone")
    print()

    t_start, n_total = time.time(), 0
    for i, (stem, stages, cap) in enumerate(videos, start=1):
        if only and i not in only:
            continue
        # one mappable per panel (a colour bar consumes the one it is given), but
        # both are built from the same cap, so the two panels stay comparable
        ticks = scale_for(cap)[1]
        sms = {term: scale_for(cap)[0] for term, _, _ in PANELS}
        path = OUT_DIR / f"{i:02d}_{stem}.mp4"
        n_vid = 0
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=CRF_QUALITY,
                                macro_block_size=MACRO_BLOCK, pixelformat="yuv420p",
                                output_params=["-preset", "slow"],
                                ffmpeg_log_level="error") as writer:
            for secs, c0, c1, w0, w1, focus, solo, marks in stages:
                n = s(secs)
                for k in range(n):
                    t = smoothstep(k / max(n - 1, 1))
                    cam = interp_cam(c0, c1, t)
                    w = tuple(a + (b - a) * t for a, b in zip(w0, w1))
                    mk = [(lv, cd, a if np.isscalar(a) else a[0] + (a[1] - a[0]) * t)
                          for lv, cd, a in marks]
                    frame = render(G, T, sms, ticks, cam, w, focus, solo, mk)
                    writer.append_data(frame)
                    if SAVE_FRAMES:
                        imageio.imwrite(frames_dir / f"{stem}_{n_vid:04d}.png", frame)
                    n_vid += 1
                    n_total += 1
        print(f"  saved {path.name}  ({n_vid} frames, {n_vid / FPS:.1f}s, "
              f"{time.time() - t_start:.0f}s elapsed)")

    print(f"\n{n_total} frames in {time.time() - t_start:.0f}s → {OUT_DIR}")


if __name__ == "__main__":
    main()
