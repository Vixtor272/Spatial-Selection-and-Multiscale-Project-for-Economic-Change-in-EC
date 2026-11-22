"""Five chained zoom videos: how Loja's flat 2023->2024 average comes apart.

The case is the one STEP F of `masked_growth_multiscale_maps.ipynb` features: a
$1,006M provincial economy whose real sales moved +0.0%, inside which 34 parishes
grew and 33 declined. Leave-one-out, the median and every dashboard test return
"honest, boring average" -- and every one of them is wrong.

The five videos are one continuous camera move, cut into chapters. Each video
starts on the exact frame the previous one ended on, so they can be played
back-to-back or shown one at a time:

  1  ecuador_to_loja      Ecuador at province scale -> zoom into Loja, the
                          province whose year reads +0.0%.
  2  loja_into_cantons    Loja dissolves into its 16 cantons: the flat average
                          splits into deep blue next to deep red.
                          (The dissolve is the device -- leave it alone.)
  3  zoom_growth_canton   Zoom into the fastest-growing canton (Pindal), which
                          then dissolves into parishes -- one parish IS the
                          canton's growth.
  4  back_to_cantons      Pull back out of Pindal, re-assemble to canton scale.
  5  zoom_decline_canton  Zoom into the biggest declining canton (Catamayo),
                          dragged down almost entirely by one parish.

Every unit at every scale is colored by the SAME quantity -- its own
Dln(real sales) over the window -- on the SAME fixed diverging scale: the Zamora
video's gradient, red = degrowth, pale blue = no change, blue = growth.
Fixing the scale across all five videos is the point: Ecuador's provinces and
Loja itself come out flat and pale, and the color only appears once you are
inside the province. A per-map scale would have hidden exactly that.

The frame carries a title, the colour bar, and place names. Nothing else: a zoom
does not need a caption saying it is a zoom, and the numbers belong in the talk,
not burnt into the picture. `main()` prints them to stdout instead.

Data note: 2024 is missing November, so every comparison against it uses the 11
months the two years share.
"""

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
Y0, Y1 = 2023, 2024               # the window; growth is Y0 -> Y1
SHARED_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]   # months present in both 2023 & 2024

PROV_CODE = "11"                  # Loja -- the province with the flat average
GROW_CANT = "1114"                # Pindal    -- canton carried by one parish upward
GROW_PARR = "111450"              # Pindal (parish)
DECL_CANT = "1116"                # Olmedo  -- canton dragged by one parish downward
DECL_PARR = "111650"              # Olmedo (La Toma)

CAP  = 0.50                       # symmetric colour cap, in log points (~ -39% / +65%)
CBAR_TICKS = [-0.50, -0.25, 0.0, 0.25, 0.50]
CBAR_LABEL = r"$\Delta\ln(\mathrm{sales})$"
# The project's two-sided map scale (ecuador_price/style.py): dark red = fall,
# light grey = no change, dark blue = rise.
CMAP = style.DIVERGING
NO_DATA = style.NO_DATA           # units with no formal sales in both years
BACKDROP = style.BACKDROP         # the rest of the country, for orientation
FPS = 30                          # 25 is what the Zamora video used; 30 pans smoother

FIG_W, FIG_H, DPI = 12.8, 7.2, 150        # 1920x1080. Font sizes and line widths are in
                                          # points, so raising DPI alone re-renders the
                                          # identical layout with 2.25x the pixels
CRF_QUALITY = 9                   # imageio 1-10; 9 -> h264 crf 5, visually lossless
MACRO_BLOCK = 8                   # 1080 is not a multiple of 16: at 16 imageio would
                                  # pad the frame to 1088 and letterbox the whole video
AX_RECT = [0.008, 0.030, 0.735, 0.845]    # the map; fixed so nothing jitters. Its
                                          # aspect (~1.55) is wider than every scene's,
                                          # so each map is height-limited and none shrinks
CBAR_RECT = [0.790, 0.190, 0.019, 0.560]

OUT_DIR = paths.STORY / "loja_zoom"
SAVE_FRAMES = False               # also dump every frame as a PNG



# --------------------------------------------------------------------------- #
# Data: one growth number per unit, at any scale, optionally inside one province
# --------------------------------------------------------------------------- #
def growth_table(panel, width, within=None):
    """One row per unit (= first `width` digits of the parish code): real sales in
    both years, mean monthly firm stock in both years, and the unit's own log
    growth. A unit needs > $1k of real sales in BOTH years to be valid."""
    d = panel[panel["year"].isin([Y0, Y1])]
    if 2024 in (Y0, Y1):
        d = d[d["month"].isin(SHARED_MONTHS)]
    d = d.assign(unit=d["cod_parr"].str[:width])
    if within is not None:
        d = d[d["unit"].str.startswith(within)]
    sales = d.groupby(["unit", "year"])["real_sales"].sum().unstack()
    firms = (d.groupby(["unit", "year", "month"])["total_firms"].sum()   # sections summed
               .groupby(["unit", "year"]).mean().unstack())              # -> mean monthly stock
    sales = sales[(sales > dec.SALES_FLOOR).all(axis=1)]
    t = pd.DataFrame({"s0": sales[Y0], "s1": sales[Y1],
                      "n0": firms[Y0].reindex(sales.index),
                      "n1": firms[Y1].reindex(sales.index)})
    return t.assign(growth=np.log(t.s1 / t.s0), delta=t.s1 - t.s0)


def aggregate_growth(t):
    """The dashboard number for a set of units: log growth of their total."""
    return float(np.log(t.s1.sum() / t.s0.sum()))


def pct(logpts):
    """Log points as the percentage a reader expects."""
    return np.expm1(logpts) * 100


# --------------------------------------------------------------------------- #
# Camera: a window in map units that covers a set of bounds at the axes' aspect
# --------------------------------------------------------------------------- #
AX_ASPECT = (FIG_W * AX_RECT[2]) / (FIG_H * AX_RECT[3])     # width / height


def camera(bounds, pad, bias=0.45):
    """(cx, cy, half_height) covering `bounds` with a `pad` fractional margin.

    Ecuador, Loja and Catamayo are all taller than a 16:9 frame, so they can only
    fill it vertically. `bias` slides the leftover width to the left, parking the
    map beside the legend instead of stranding it in the middle of the frame."""
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
NORM = Normalize(vmin=-CAP, vmax=CAP)
SM = ScalarMappable(norm=NORM, cmap=CMAP)
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]


NO_DATA_RGBA = to_rgba(NO_DATA)


def colors_for(geo, key, table):
    """Per-polygon fill: the unit's growth on the fixed scale, grey if not in the
    table (no formal sales in both years). All RGBA -- geopandas needs one dtype."""
    g = geo[key].map(table["growth"])
    return [NO_DATA_RGBA if pd.isna(v) else SM.to_rgba(v) for v in g]


def draw_layer(ax, geo, key, table, alpha, lw=0.5, edge="white"):
    if alpha <= 0.002 or geo.empty:
        return
    geo.plot(color=colors_for(geo, key, table), ax=ax, alpha=alpha,
             edgecolor=edge, linewidth=lw)


def wrap(name, width=13):
    """Break a long name across lines so its label stays inside its own polygon --
    'San Pedro De La Bendita' laid out flat runs across three of its neighbours."""
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
# One frame
# --------------------------------------------------------------------------- #
def render(G, T, cam, w_prov, w_cant, w_parr, focus_cant, marks):
    """`marks` = list of (level, code, alpha) black outlines, already resolved to
    this frame's alpha. Layer weights fade layers in and out; whatever sits below
    a fading layer stays opaque, so nothing ever muddies."""
    cx, cy, half = cam
    view = ((cx - half * AX_ASPECT, cx + half * AX_ASPECT), (cy - half, cy + half))

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes(AX_RECT)

    # the rest of the country, always, so you never lose your bearings
    G["prov"].plot(color=BACKDROP, ax=ax, edgecolor="white", linewidth=0.4, zorder=0)

    draw_layer(ax, G["prov"], "DPA_PROVIN", T["prov"], w_prov, lw=0.4)
    draw_layer(ax, G["cant"], "DPA_CANTON", T["cant"], w_cant, lw=0.6)
    if w_parr > 0 and focus_cant is not None:
        draw_layer(ax, G["parr"][G["parr"]["DPA_CANTON"] == focus_cant],
                   "DPA_PARROQ", T["parr"], w_parr, lw=0.6)

    for level, code, a in marks:
        key = {"prov": "DPA_PROVIN", "cant": "DPA_CANTON", "parr": "DPA_PARROQ"}[level]
        outline(ax, G[level], key, code, a, lw=2.4 if level == "parr" else 2.0)

    # provinces go unnamed -- the outline and the zoom already say which one it is
    star = next((c for lv, c, a in marks if lv == "parr" and a > 0.5), None)
    draw_labels(ax, G["cant"], "DPA_CANTON", NAMES["cant"], w_cant * (1 - w_parr), view)
    if focus_cant is not None:
        draw_labels(ax, G["parr"][G["parr"]["DPA_CANTON"] == focus_cant],
                    "DPA_PARROQ", NAMES["parr"], w_parr, view, fontsize=8.5, big=star)

    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.022, 0.940, TITLE, fontsize=16, fontweight="bold", color="#111")

    cax = fig.add_axes(CBAR_RECT)
    cb = fig.colorbar(SM, cax=cax, ticks=CBAR_TICKS, extend="both")
    cb.ax.set_yticklabels([f"{v:+.2f}   ({pct(v):+.0f}%)" for v in CBAR_TICKS], fontsize=9.5)
    cb.outline.set_visible(False)
    cax.tick_params(length=2.5)
    x, y, _, h = CBAR_RECT
    fig.text(x, y + h + 0.035, CBAR_LABEL, fontsize=10.5, color="0.25")

    canvas = fig.canvas
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global NAMES, TITLE

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = dec.load_real_panel()
    panel["month"] = panel["mes"].astype(int)

    # --- the three tables the maps are colored by ---------------------------
    T = {"prov": growth_table(panel, 2),
         "cant": growth_table(panel, 4, within=PROV_CODE),
         "parr": growth_table(panel, 6, within=PROV_CODE)}

    # --- geometry -----------------------------------------------------------
    prov = gpd.read_file(paths.SHAPEFILES["province"])
    cant = gpd.read_file(paths.SHAPEFILES["canton"])
    parr = gpd.read_file(paths.SHAPEFILES["parish"])
    G = {"prov": prov,
         "cant": cant[cant["DPA_PROVIN"] == PROV_CODE].copy(),
         "parr": parr[parr["DPA_PROVIN"] == PROV_CODE].copy()}
    NAMES = {"prov": {r.DPA_PROVIN: r.DPA_DESPRO.title() for _, r in prov.iterrows()},
             "cant": {r.DPA_CANTON: r.DPA_DESCAN.title() for _, r in G["cant"].iterrows()},
             "parr": {r.DPA_PARROQ: r.DPA_DESPAR.title().split(" (")[0]
                      for _, r in G["parr"].iterrows()}}

    # --- the numbers behind the story: printed, not drawn on the frames ------
    prov_row = T["prov"].loc[PROV_CODE]
    n_grow = int((T["parr"].growth > 0).sum())
    n_decl = int((T["parr"].growth < 0).sum())
    cant_hi = T["cant"].growth.idxmax()
    cant_lo = T["cant"].growth.idxmin()

    def canton_story(cc, pc):
        """The canton exactly as the map colors it, and what is left of it once the
        one parish `pc` is taken out. Both sides come from the same two tables the
        maps are drawn from, so the printed numbers match what you see."""
        can, par = T["cant"].loc[cc], T["parr"].loc[pc]
        rest0, rest1 = can.s0 - par.s0, can.s1 - par.s1
        sub = T["parr"][T["parr"].index.str[:4] == cc]
        return dict(G=can.growth, G_wo=np.log(rest1 / rest0), s0=can.s0, s1=can.s1,
                    share=par.delta / can.delta, row=par,
                    n=int((G["parr"].DPA_CANTON == cc).sum()),   # parishes on the map
                    n_valid=len(sub), n_grow=int((sub.growth > 0).sum()))

    up, dn = canton_story(GROW_CANT, GROW_PARR), canton_story(DECL_CANT, DECL_PARR)

    TITLE = f"Real-sales log growth ({Y0} - {Y1})"

    # the story in numbers, for the talk -- deliberately not burnt into the frames
    print(f"Ecuador {Y0}→{Y1}: {pct(aggregate_growth(T['prov'])):+.1f}% over "
          f"{len(T['prov'])} provinces")
    print(f"{NAMES['prov'][PROV_CODE]}: ${prov_row.s0/1e6:,.0f}M → ${prov_row.s1/1e6:,.0f}M   "
          f"Δln {prov_row.growth:+.4f} ({pct(prov_row.growth):+.1f}%)   "
          f"{n_grow} parishes grew, {n_decl} declined")
    print(f"  {len(T['cant'])} cantons, from {NAMES['cant'][cant_lo]} "
          f"{pct(T['cant'].growth[cant_lo]):+.0f}% to {NAMES['cant'][cant_hi]} "
          f"{pct(T['cant'].growth[cant_hi]):+.0f}%")
    for lab, cc, pcode, st in [("growth", GROW_CANT, GROW_PARR, up),
                               ("decline", DECL_CANT, DECL_PARR, dn)]:
        print(f"  {lab:8s} {NAMES['cant'][cc]:<10s} "
              f"${st['s0']/1e6:,.1f}M → ${st['s1']/1e6:,.1f}M  {pct(st['G']):+8.1f}%  |  "
              f"without {NAMES['parr'][pcode]}: {pct(st['G_wo']):+7.1f}%  |  "
              f"that parish {st['share']:+.0%} of the canton's $ change "
              f"(${st['row'].delta/1e6:+,.1f}M, {st['row'].n0:.0f}→{st['row'].n1:.0f} firms), "
              f"{st['n_grow']}/{st['n_valid']} parishes with data grew")

    # --- cameras ------------------------------------------------------------
    CAM_EC   = camera(prov.total_bounds, pad=0.03)
    CAM_LOJA = camera(prov[prov.DPA_PROVIN == PROV_CODE].total_bounds, pad=0.10)
    CAM_UP   = camera(G["cant"][G["cant"].DPA_CANTON == GROW_CANT].total_bounds, pad=0.30)
    CAM_DN   = camera(G["cant"][G["cant"].DPA_CANTON == DECL_CANT].total_bounds, pad=0.30)

    s = lambda sec: max(int(round(sec * FPS)), 1)
    MARK_LOJA = ("prov", PROV_CODE, 1.0)

    # --- the timeline: (video, seconds, cam0, cam1, weights0, weights1,
    #                    focus canton, marks) ------------------------------
    # weights = (province layer, Loja's cantons, the focus canton's parishes).
    # A mark's alpha may be a (from, to) pair, so outlines fade in and out too.
    # No captions: a hold is a beat, a zoom explains itself.
    P, C, R = (1, 0, 0), (0, 1, 0), (0, 1, 1)

    stages = [
        # ---- video 1: Ecuador -> Loja --------------------------------------
        (1, 1.6, CAM_EC, CAM_EC, P, P, None, [MARK_LOJA]),
        (1, 3.6, CAM_EC, CAM_LOJA, P, P, None, [MARK_LOJA]),
        (1, 1.6, CAM_LOJA, CAM_LOJA, P, P, None, [MARK_LOJA]),

        # ---- video 2: Loja -> its cantons (the reveal) ----------------------
        (2, 1.2, CAM_LOJA, CAM_LOJA, P, P, None, [MARK_LOJA]),
        (2, 2.4, CAM_LOJA, CAM_LOJA, P, C, None, [MARK_LOJA]),
        (2, 3.2, CAM_LOJA, CAM_LOJA, C, C, None, [MARK_LOJA]),

        # ---- video 3: into the canton that is one growing parish ------------
        (3, 1.4, CAM_LOJA, CAM_LOJA, C, C, None,
         [MARK_LOJA, ("cant", GROW_CANT, (0.0, 1.0))]),
        (3, 3.4, CAM_LOJA, CAM_UP, C, C, None,
         [("prov", PROV_CODE, (1.0, 0.0)), ("cant", GROW_CANT, 1.0)]),
        (3, 1.0, CAM_UP, CAM_UP, C, C, None, [("cant", GROW_CANT, 1.0)]),
        (3, 2.2, CAM_UP, CAM_UP, C, R, GROW_CANT,
         [("cant", GROW_CANT, (1.0, 0.35)), ("parr", GROW_PARR, (0.0, 1.0))]),
        (3, 2.6, CAM_UP, CAM_UP, R, R, GROW_CANT,
         [("cant", GROW_CANT, 0.35), ("parr", GROW_PARR, 1.0)]),

        # ---- video 4: back out to canton scale ------------------------------
        (4, 1.0, CAM_UP, CAM_UP, R, R, GROW_CANT,
         [("cant", GROW_CANT, 0.35), ("parr", GROW_PARR, 1.0)]),
        (4, 1.6, CAM_UP, CAM_UP, R, C, GROW_CANT,
         [("cant", GROW_CANT, (0.35, 1.0)), ("parr", GROW_PARR, (1.0, 0.0))]),
        (4, 3.4, CAM_UP, CAM_LOJA, C, C, None,
         [("prov", PROV_CODE, (0.0, 1.0)), ("cant", GROW_CANT, 1.0)]),
        (4, 1.8, CAM_LOJA, CAM_LOJA, C, C, None,
         [MARK_LOJA, ("cant", GROW_CANT, (1.0, 0.0)), ("cant", DECL_CANT, (0.0, 1.0))]),

        # ---- video 5: into the canton that is one declining parish ----------
        (5, 1.4, CAM_LOJA, CAM_LOJA, C, C, None, [MARK_LOJA, ("cant", DECL_CANT, 1.0)]),
        (5, 3.4, CAM_LOJA, CAM_DN, C, C, None,
         [("prov", PROV_CODE, (1.0, 0.0)), ("cant", DECL_CANT, 1.0)]),
        (5, 1.0, CAM_DN, CAM_DN, C, C, None, [("cant", DECL_CANT, 1.0)]),
        (5, 2.2, CAM_DN, CAM_DN, C, R, DECL_CANT,
         [("cant", DECL_CANT, (1.0, 0.35)), ("parr", DECL_PARR, (0.0, 1.0))]),
        (5, 2.8, CAM_DN, CAM_DN, R, R, DECL_CANT,
         [("cant", DECL_CANT, 0.35), ("parr", DECL_PARR, 1.0)]),
    ]

    # --- render, one video per chapter, each starting where the last ended ---
    titles = {1: "1_ecuador_to_loja", 2: "2_loja_into_cantons",
              3: "3_zoom_growth_canton", 4: "4_back_to_cantons",
              5: "5_zoom_decline_canton"}
    frames_dir = OUT_DIR / "frames"
    if SAVE_FRAMES:
        frames_dir.mkdir(exist_ok=True)

    t_start, n_total = time.time(), 0
    for vid in sorted(titles):
        path = OUT_DIR / f"{titles[vid]}.mp4"
        n_vid = 0
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=CRF_QUALITY,
                                macro_block_size=MACRO_BLOCK, pixelformat="yuv420p",
                                output_params=["-preset", "slow"],
                                ffmpeg_log_level="error") as writer:
            for stage in stages:
                v, secs, c0, c1, w0, w1, focus, marks = stage
                if v != vid:
                    continue
                n = s(secs)
                for k in range(n):
                    t = smoothstep(k / max(n - 1, 1))
                    cam = interp_cam(c0, c1, t)
                    w = tuple(a + (b - a) * t for a, b in zip(w0, w1))
                    mk = [(lv, cd, a if np.isscalar(a) else a[0] + (a[1] - a[0]) * t)
                          for lv, cd, a in marks]
                    frame = render(G, T, cam, *w, focus, mk)
                    writer.append_data(frame)
                    if SAVE_FRAMES:
                        imageio.imwrite(frames_dir / f"v{vid}_{n_vid:04d}.png", frame)
                    n_vid += 1
                    n_total += 1
        print(f"  saved {path.name}  ({n_vid} frames, {n_vid / FPS:.1f}s)")

    print(f"\n{n_total} frames in {time.time() - t_start:.0f}s → {OUT_DIR}")


if __name__ == "__main__":
    main()
