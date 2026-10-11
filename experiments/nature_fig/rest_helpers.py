"""Placement helpers for the fig2/fig5/fig6 candidates (layout only; no data logic).

* ``place_text``: puts a label in the free space of an axes. It rasterises every mark
  already drawn (lines, markers, collections, patches, texts) into an occupancy grid at
  final physical size and picks the free position nearest a preferred anchor, so label
  positions follow the data instead of being hard-coded.
* ``swarm``: deterministic beeswarm offsets (no random jitter) so overlaid points do not
  hide each other.
* ``nice_ticks``: sparse, round tick values that always include zero.
"""
import numpy as np
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.text import Text

PT = 72.0


def _px_per_pt(fig):
    return fig.dpi / PT


def occupancy(ax, pad_pt=1.2, res_pt=0.5, exclude=()):
    """Boolean grid (rows = y, cols = x) of the axes area marking drawn content.

    Coordinates are display pixels; the grid origin is the axes' lower-left corner.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = _px_per_pt(fig)
    bb = ax.get_window_extent(r)
    res = res_pt * k
    nx = int(np.ceil(bb.width / res)) + 1
    ny = int(np.ceil(bb.height / res)) + 1
    grid = np.zeros((ny, nx), dtype=bool)
    pad = pad_pt * k

    def mark_disc(xs, ys, rad):
        rad = rad + pad
        rr = int(np.ceil(rad / res))
        for x, y in zip(xs, ys):
            cx = int(round((x - bb.x0) / res))
            cy = int(round((y - bb.y0) / res))
            x0, x1 = max(cx - rr, 0), min(cx + rr + 1, nx)
            y0, y1 = max(cy - rr, 0), min(cy + rr + 1, ny)
            if x0 < x1 and y0 < y1:
                grid[y0:y1, x0:x1] = True

    def mark_rect(x0, y0, x1, y1):
        c0 = max(int(np.floor((x0 - pad - bb.x0) / res)), 0)
        c1 = min(int(np.ceil((x1 + pad - bb.x0) / res)) + 1, nx)
        r0 = max(int(np.floor((y0 - pad - bb.y0) / res)), 0)
        r1 = min(int(np.ceil((y1 + pad - bb.y0) / res)) + 1, ny)
        if c0 < c1 and r0 < r1:
            grid[r0:r1, c0:c1] = True

    def mark_polyline(xy, lw_px):
        if len(xy) < 2:
            if len(xy) == 1:
                mark_disc([xy[0][0]], [xy[0][1]], lw_px / 2)
            return
        pts = []
        for (xa, ya), (xb, yb) in zip(xy[:-1], xy[1:]):
            if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
                continue
            n = max(int(np.hypot(xb - xa, yb - ya) / (res * 0.5)), 1)
            t = np.linspace(0, 1, n + 1)
            pts.append(np.column_stack([xa + (xb - xa) * t, ya + (yb - ya) * t]))
        if pts:
            p = np.vstack(pts)
            mark_disc(p[:, 0], p[:, 1], lw_px / 2)

    for art in ax.get_children():
        if art in exclude or not art.get_visible():
            continue
        if isinstance(art, Line2D):
            xy = art.get_transform().transform(art.get_xydata())
            # clip to the axes
            if art.get_linestyle() not in ("None", "none", "", " ") and art.get_linewidth() > 0:
                mark_polyline(xy, art.get_linewidth() * k)
            if art.get_marker() not in ("None", "none", "", " ", None):
                rad = (art.get_markersize() / 2 + art.get_markeredgewidth()) * k
                mark_disc(xy[:, 0], xy[:, 1], rad)
        elif isinstance(art, PathCollection):
            offs = art.get_offset_transform().transform(art.get_offsets())
            sizes = art.get_sizes()
            if len(sizes) == 0:
                continue
            rad = np.sqrt(np.max(sizes)) / 2 * k + np.max(art.get_linewidths() if len(art.get_linewidths()) else [0]) * k
            mark_disc(offs[:, 0], offs[:, 1], rad)
        elif isinstance(art, LineCollection):
            lws = art.get_linewidths()
            lw = (np.max(lws) if len(lws) else 1.0) * k
            for seg in art.get_segments():
                mark_polyline(art.get_transform().transform(seg), lw)
        elif isinstance(art, (Rectangle, PathPatch)):
            if art is ax.patch:
                continue
            e = art.get_window_extent(r)
            mark_rect(e.x0, e.y0, e.x1, e.y1)
        elif isinstance(art, Text):
            if not art.get_text().strip():
                continue
            e = art.get_window_extent(r)
            mark_rect(e.x0, e.y0, e.x1, e.y1)
    return grid, bb, res


def place_text(ax, text, prefer, fontsize, color, ha="left", va="bottom", region=None,
               pad_pt=1.2, edge_pt=2.0, linespacing=1.15, fontweight="normal", exclude=(),
               zorder=6, weights=(1.0, 1.0)):
    """Place ``text`` in free space nearest ``prefer`` (axes fraction of the anchor).

    ``region`` = (x0, y0, x1, y1) in axes fraction limits where the text box may sit.
    ``weights`` scale the horizontal and vertical distance to ``prefer`` (a line label uses
    a large horizontal weight so it stays beside its line at whatever height is free).
    Returns the Text artist. Raises RuntimeError if no free position exists.
    """
    fig = ax.figure
    grid, bb, res = occupancy(ax, pad_pt=pad_pt, exclude=exclude)
    r = fig.canvas.get_renderer()
    probe = ax.text(0, 0, text, fontsize=fontsize, ha="left", va="bottom",
                    linespacing=linespacing, fontweight=fontweight, transform=ax.transAxes)
    e = probe.get_window_extent(r)
    probe.remove()
    w, h = e.width, e.height
    k = _px_per_pt(fig)
    edge = edge_pt * k
    ny, nx = grid.shape
    sat = np.zeros((ny + 1, nx + 1), dtype=np.int32)
    sat[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    wc = int(np.ceil(w / res))
    hc = int(np.ceil(h / res))
    rx0, ry0, rx1, ry1 = region if region is not None else (0, 0, 1, 1)
    c_lo = max(int(np.ceil((rx0 * bb.width + edge) / res)), 0)
    c_hi = min(int(np.floor((rx1 * bb.width - edge) / res)) - wc, nx - wc - 1)
    r_lo = max(int(np.ceil((ry0 * bb.height + edge) / res)), 0)
    r_hi = min(int(np.floor((ry1 * bb.height - edge) / res)) - hc, ny - hc - 1)
    if c_hi < c_lo or r_hi < r_lo:
        raise RuntimeError(f"label {text!r} does not fit the region")
    cs = np.arange(c_lo, c_hi + 1)
    rs = np.arange(r_lo, r_hi + 1)
    C, R = np.meshgrid(cs, rs)
    occ = sat[R + hc, C + wc] - sat[R, C + wc] - sat[R + hc, C] + sat[R, C]
    free = occ == 0
    if not free.any():
        raise RuntimeError(f"no free position for label {text!r}")
    # anchor of the candidate box in axes fraction, matching ha/va
    fx = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
    fy = {"bottom": 0.0, "center": 0.5, "top": 1.0}[va]
    ax_x = (C * res + fx * w) / bb.width
    ax_y = (R * res + fy * h) / bb.height
    d = (weights[0] * (ax_x - prefer[0]) * bb.width) ** 2 + (weights[1] * (ax_y - prefer[1]) * bb.height) ** 2
    d = np.where(free, d, np.inf)
    i = np.unravel_index(np.argmin(d), d.shape)
    t = ax.text(ax_x[i], ax_y[i], text, fontsize=fontsize, color=color, ha=ha, va=va,
                linespacing=linespacing, fontweight=fontweight, transform=ax.transAxes,
                zorder=zorder)
    return t


def swarm(values_px, diameter_px, max_half_px):
    """Deterministic beeswarm: perpendicular offsets (px) so markers do not overlap.

    Points are placed in order of value; each takes the smallest |offset| (alternating
    sides) that clears every placed neighbour. If the band is full, the least-overlapping
    slot inside the band is used.
    """
    v = np.asarray(values_px, dtype=float)
    order = np.argsort(v, kind="stable")
    placed_v, placed_o = [], []
    out = np.zeros_like(v)
    step = diameter_px * 0.5
    nmax = int(max_half_px // step)
    slots = [0.0]
    for j in range(1, nmax + 1):
        slots += [j * step, -j * step]
    for idx in order:
        best, best_cost = None, np.inf
        for o in slots:
            cost = 0.0
            ok = True
            for pv, po in zip(placed_v[::-1], placed_o[::-1]):
                if v[idx] - pv >= diameter_px:
                    break
                dist = np.hypot(v[idx] - pv, o - po)
                if dist < diameter_px * 0.98:
                    ok = False
                    cost += diameter_px - dist
            if ok:
                best = o
                break
            if cost < best_cost:
                best_cost, best = cost, o
        out[idx] = best
        # keep placed lists sorted by value (input is processed in value order)
        placed_v.append(v[idx])
        placed_o.append(best)
    return out


def nice_ticks(lo, hi, nmin=3, nmax=4, steps=(1, 2, 2.5, 4, 5, 10, 20)):
    """Round tick values covering [lo, hi] (data range), always including 0."""
    best = None
    for s in steps:
        t0 = np.ceil((lo - 1e-9) / s) * s
        ticks = np.arange(t0, hi + 1e-9, s)
        ticks = [round(float(x), 6) for x in ticks]
        if 0.0 not in ticks and lo <= 0 <= hi:
            continue
        if nmin <= len(ticks) <= nmax:
            best = ticks
            break
    if best is None:
        s = steps[-1]
        best = [round(float(x), 6) for x in np.arange(np.floor(lo / s) * s, hi + 1e-9, s)]
    return best
