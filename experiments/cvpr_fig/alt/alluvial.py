"""Minimal alluvial (two-stage Sankey) drawing for the figure candidates: nodes are stacked bars at
x = 0 and x = 1, flows are ribbons with cubic Bezier sides. Values are shares (any unit)."""
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path


def ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=1.0, z=2):
    """Ribbon from the left interval [y0a, y0b] at x0 to the right interval [y1a, y1b] at x1."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),       # lower edge
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def alluvial(ax, left, right, flows, colors, gap=0.04, w=0.06, node_color="#4D4D4D", x1=1.0):
    """left/right: ordered lists of node names (top to bottom); flows: {(l, r): value};
    colors: {(l, r): colour}. Returns {node: (y0, y1)} for both sides, top of the axes at 1."""
    tot = sum(flows.values())
    def stack(nodes, side):
        pos, y = {}, 1.0
        n_gaps = len(nodes) - 1
        scale = (1.0 - gap * n_gaps) / tot
        for nd in nodes:
            v = sum(val for (l, r), val in flows.items() if (l if side == 0 else r) == nd)
            pos[nd] = (y - v * scale, y)
            y -= v * scale + gap
        return pos, scale
    lp, sc = stack(left, 0)
    rp, _ = stack(right, 1)
    lcur = {nd: lp[nd][1] for nd in left}
    rcur = {nd: rp[nd][1] for nd in right}
    for l in left:                      # flows leave each left node in the order of the right nodes
        for r in right:
            v = flows.get((l, r), 0.0)
            if v <= 0:
                continue
            h = v * sc
            ribbon(ax, w, lcur[l] - h, lcur[l], x1 - w, rcur[r] - h, rcur[r], colors[(l, r)])
            lcur[l] -= h
            rcur[r] -= h
    for nd, (a, b) in lp.items():
        ax.add_patch(Rectangle((0, a), w, b - a, facecolor=node_color, edgecolor="none", zorder=3))
    for nd, (a, b) in rp.items():
        ax.add_patch(Rectangle((x1 - w, a), w, b - a, facecolor=node_color, edgecolor="none", zorder=3))
    ax.set_xlim(-0.02, x1 + 0.02)
    ax.set_ylim(-0.02, 1.02)
    return lp, rp
