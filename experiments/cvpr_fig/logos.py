"""Model logos for the CVPR figures, drawn as vector paths on a millimetre axes.

Source: lobe-icons static SVGs (@lobehub/icons-static-svg 1.95.1, MIT licence), stored in
experiments/cvpr_fig/logos/. The marks identify the model families (OpenAI for GPT-5 mini, Gemini,
Claude, Qwen) and remain trademarks of their owners.

draw(ax, model, cx, cy, h) draws the logo of `model` with height h, centred on (cx, cy), in the data
coordinates of `ax`, which must have equal x and y units (the figures' millimetre overlay). Solid fills
become PathPatch objects; SVG linear gradients become small RGBA images clipped to the vector path, so
the outline stays vector at any print size.
"""
import pathlib
import re

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.path import Path

DIR = pathlib.Path(__file__).resolve().parent / "logos"
FILE = {"gpt-5-mini": "openai.svg", "gemini-3.7-flash": "gemini-color.svg",
        "claude-haiku-4-5-20251001": "claude-color.svg", "qwen3.8-flash": "qwen-color.svg"}
INK = "#1F1F1F"                    # fill for marks drawn in currentColor (OpenAI)


_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def _tokens(d):
    """Commands with their numbers; arc flags are single characters (minified SVGs write "0 01.5.5")."""
    i, n, out = 0, len(d), []
    while i < n:
        c = d[i]
        if c.isalpha():
            cmd, i, args = c, i + 1, []
            k = 0
            while True:
                while i < n and d[i] in " ,\t\n\r":
                    i += 1
                if i >= n or d[i].isalpha():
                    break
                if cmd.upper() == "A" and k % 7 in (3, 4):
                    args.append(float(d[i])); i += 1
                else:
                    m = _NUM.match(d, i)
                    args.append(float(m.group(0))); i = m.end()
                k += 1
            out.append((cmd, args))
        else:
            i += 1
    return out


def _arc(p0, rx, ry, phi, fa, fs, p1):
    """SVG endpoint arc -> cubic Bezier control points (list of (c1, c2, end))."""
    if rx == 0 or ry == 0 or np.allclose(p0, p1):
        return [(p0, p1, p1)]
    rx, ry = abs(rx), abs(ry)
    cp, sp = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    dx, dy = (p0 - p1) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * np.sqrt(lam), ry * np.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    co = np.sqrt(max(num, 0) / (rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2)) * (-1 if fa == fs else 1)
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    c = np.array([cp * cxp - sp * cyp, sp * cxp + cp * cyp]) + (p0 + p1) / 2
    ang = lambda u, v: np.arctan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])
    u = np.array([(x1p - cxp) / rx, (y1p - cyp) / ry])
    v = np.array([(-x1p - cxp) / rx, (-y1p - cyp) / ry])
    t0, dt = ang(np.array([1.0, 0.0]), u), ang(u, v)
    if not fs and dt > 0:
        dt -= 2 * np.pi
    elif fs and dt < 0:
        dt += 2 * np.pi
    segs, out = int(np.ceil(abs(dt) / (np.pi / 2))), []
    R = np.array([[cp, -sp], [sp, cp]])
    for k in range(segs):
        a0, a1 = t0 + dt * k / segs, t0 + dt * (k + 1) / segs
        al = 4 / 3 * np.tan((a1 - a0) / 4)
        e0, e1 = np.array([np.cos(a0), np.sin(a0)]), np.array([np.cos(a1), np.sin(a1)])
        d0, d1 = np.array([-np.sin(a0), np.cos(a0)]), np.array([-np.sin(a1), np.cos(a1)])
        f = lambda q: c + R @ (q * [rx, ry])
        out.append((f(e0 + al * d0), f(e1 - al * d1), f(e1)))
    return out


def parse_path(d):
    verts, codes = [], []
    cur = start = np.zeros(2)
    last_c = last_q = None
    for cmd, a in _tokens(d):
        C, rel = cmd.upper(), cmd.islower()
        if C == "Z":
            verts.append(start); codes.append(Path.CLOSEPOLY); cur = start; last_c = last_q = None
            continue
        k = _ARITY[C]
        for j in range(0, len(a), k):
            q = a[j:j + k]
            off = cur if rel else np.zeros(2)
            if C == "M":
                cur = start = off + q[:2]; verts.append(cur); codes.append(Path.MOVETO)
                C = "L"; last_c = last_q = None          # further pairs after M are line-tos
                continue
            if C in "LHV":
                if C == "L":
                    cur = off + q[:2]
                elif C == "H":
                    cur = np.array([(cur[0] if rel else 0) + q[0], cur[1]])
                else:
                    cur = np.array([cur[0], (cur[1] if rel else 0) + q[0]])
                verts.append(cur); codes.append(Path.LINETO); last_c = last_q = None
            elif C in "CS":
                if C == "C":
                    c1, c2, e = off + q[0:2], off + q[2:4], off + q[4:6]
                else:
                    c1 = 2 * cur - last_c if last_c is not None else cur
                    c2, e = off + q[0:2], off + q[2:4]
                verts += [c1, c2, e]; codes += [Path.CURVE4] * 3; last_c, last_q, cur = c2, None, e
            elif C in "QT":
                if C == "Q":
                    c1, e = off + q[0:2], off + q[2:4]
                else:
                    c1 = 2 * cur - last_q if last_q is not None else cur
                    e = off + q[0:2]
                verts += [c1, e]; codes += [Path.CURVE3] * 2; last_q, last_c, cur = c1, None, e
            elif C == "A":
                e = off + q[5:7]
                for c1, c2, end in _arc(cur, q[0], q[1], q[2], q[3], q[4], e):
                    verts += [c1, c2, end]; codes += [Path.CURVE4] * 3
                cur, last_c, last_q = e, None, None
    return Path(np.array(verts, float), codes)


def _num(v):
    v = v.strip()
    return float(v[:-1]) / 100 if v.endswith("%") else float(v)


def _gradients(svg):
    out = {}
    for m in re.finditer(r'<linearGradient([^>]*)>(.*?)</linearGradient>', svg, re.S):
        a = m.group(1)
        get = lambda k, d=None: (re.search(rf'\b{k}="([^"]+)"', a).group(1) if re.search(rf'\b{k}="', a) else d)
        stops = []
        for s in re.finditer(r'<stop([^>]*)/?>', m.group(2)):
            sa = s.group(1)
            off = re.search(r'offset="([^"]+)"', sa)
            op = re.search(r'stop-opacity="([^"]+)"', sa)
            stops.append((_num(off.group(1)) if off else 0.0, re.search(r'stop-color="([^"]+)"', sa).group(1),
                          float(op.group(1)) if op else 1.0))
        out[get("id")] = {"units": get("gradientUnits", "objectBoundingBox"), "x1": get("x1", "0"), "y1": get("y1", "0"),
                          "x2": get("x2", "100%"), "y2": get("y2", "0"), "stops": stops}
    return out


def layers(model):
    """[(matplotlib Path in SVG units, fill)]: fill is a colour string or a gradient dict."""
    svg = (DIR / FILE[model]).read_text()
    grads = _gradients(svg)
    out = []
    for m in re.finditer(r'<path([^>]*)/?>', svg):
        a = m.group(1)
        d = re.search(r'\bd="([^"]+)"', a).group(1)
        f = re.search(r'\bfill="([^"]+)"', a)
        fill = f.group(1) if f else INK
        if fill.startswith("url(#"):
            fill = grads[fill[5:-1]]
        elif fill in ("currentColor", "none"):
            fill = INK
        out.append((parse_path(d), fill))
    return out


def _gradient_image(path, g, n=160):
    (x0, y0), (x1, y1) = path.get_extents().get_points()
    if g["units"] == "userSpaceOnUse":
        gx1, gy1, gx2, gy2 = (_num(g[k]) for k in ("x1", "y1", "x2", "y2"))
    else:                                           # objectBoundingBox: fractions of the path's box
        gx1, gx2 = x0 + _num(g["x1"]) * (x1 - x0), x0 + _num(g["x2"]) * (x1 - x0)
        gy1, gy2 = y0 + _num(g["y1"]) * (y1 - y0), y0 + _num(g["y2"]) * (y1 - y0)
    u, v = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))      # row 0 = smallest SVG y (top)
    dx, dy = gx2 - gx1, gy2 - gy1
    t = np.clip(((u - gx1) * dx + (v - gy1) * dy) / max(dx * dx + dy * dy, 1e-12), 0, 1)
    offs = np.array([s[0] for s in g["stops"]])
    rgb = np.array([to_rgb(s[1]) for s in g["stops"]])
    alpha = np.array([s[2] for s in g["stops"]])
    img = np.zeros((n, n, 4))
    for c in range(3):
        img[..., c] = np.interp(t, offs, rgb[:, c])
    img[..., 3] = np.interp(t, offs, alpha)
    return img, (x0, x1, y1, y0)


def draw(ax, model, cx, cy, h, zorder=6):
    T = Affine2D().translate(-12, -12).scale(h / 24, -h / 24).translate(cx, cy) + ax.transData
    for path, fill in layers(model):
        if isinstance(fill, str):
            ax.add_patch(PathPatch(path, transform=T, fc=fill, ec="none", lw=0, zorder=zorder))
        else:
            img, ext = _gradient_image(path, fill)
            im = ax.imshow(img, extent=ext, transform=T, interpolation="bilinear", zorder=zorder, aspect="auto")
            im.set_clip_path(PathPatch(path, transform=T))
