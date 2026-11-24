"""Whiteboard video engine.

lesson steps  ->  layout every board item (text, math, boxes, flows, tables, page images)
              ->  plan: when each item is written, synchronised with the narration length
              ->  render: frames (items are "written" left-to-right with a marker) piped to ffmpeg
"""
import glob
import math
import os
import re
import subprocess
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from common import CACHE, WORK, env, log

SR = 24000

# ============================================================================ fonts
FONT_DIR = CACHE / "fonts"
FONT_URLS = {
    "latin": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf",
    ],
    "arabic": [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/tajawal/Tajawal-Medium.ttf",
    ],
}
SYSTEM_GLOBS = {
    "latin": ["**/PatrickHand*.ttf", "**/ComicNeue-Regular*.ttf", "**/Kalam-Regular*.ttf", "**/Caveat*.ttf"],
    "arabic": ["**/Amiri-Regular*.ttf", "**/NotoNaskhArabic*.ttf", "**/NotoSansArabic*.ttf", "**/Tajawal*Medium*.ttf"],
}


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=40) as r:
            data = r.read()
        if len(data) < 20000 or data[:4] not in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        log(f"    font download failed: {url.rsplit('/', 1)[-1]} ({type(e).__name__})")
        return False


def find_font(kind: str):
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(FONT_DIR.glob(f"{kind}-*.ttf")):
        return str(f)
    for url in FONT_URLS[kind]:
        dest = FONT_DIR / f"{kind}-{url.rsplit('/', 1)[-1]}"
        if _download(url, dest):
            return str(dest)
    for base in ("/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts")):
        for pat in SYSTEM_GLOBS[kind]:
            hits = sorted(glob.glob(os.path.join(base, pat), recursive=True))
            if hits:
                return hits[0]
    return None


def setup_fonts() -> list:
    fams = []
    for kind in ("latin", "arabic"):
        p = find_font(kind)
        if not p:
            log(f"  font[{kind}]: nothing found, DejaVu Sans will be used")
            continue
        try:
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            fams.append(name)
            log(f"  font[{kind}] = {name}")
        except Exception as e:
            log(f"  font {p} unusable: {e}")
    fams.append("DejaVu Sans")
    return list(dict.fromkeys(fams))


# ============================================================================ themes
THEMES = {
    "chalk": {
        "bg": (36, 62, 52), "bg2": (22, 40, 34), "frame": (101, 70, 42),
        "colors": {"white": (242, 242, 232), "yellow": (255, 227, 112), "blue": (140, 206, 255),
                   "green": (152, 236, 168), "pink": (255, 152, 178), "orange": (255, 192, 112)},
    },
    "white": {
        "bg": (252, 252, 248), "bg2": (232, 232, 226), "frame": (165, 165, 172),
        "colors": {"white": (30, 30, 38), "yellow": (190, 112, 0), "blue": (20, 88, 200),
                   "green": (14, 128, 58), "pink": (205, 35, 85), "orange": (214, 92, 10)},
    },
}


def make_background(W, H, theme: str) -> np.ndarray:
    t = THEMES[theme]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    k = np.clip(1 - 0.28 * d, 0, 1)[..., None]
    c1, c2 = np.array(t["bg"], np.float32), np.array(t["bg2"], np.float32)
    bg = c2 + (c1 - c2) * k
    bg += np.random.default_rng(7).normal(0, 1.5, (H, W, 1)).astype(np.float32)
    img = np.clip(bg, 0, 255).astype(np.uint8)
    b = max(6, int(H * 0.012))
    img[:b] = img[-b:] = t["frame"]
    img[:, :b] = img[:, -b:] = t["frame"]
    return img


# ============================================================================ text
AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
TOKEN_RE = re.compile(r"(?:\$[^$]*\$|[^\s$])+")
_SYM = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "mu": "µ", "pi": "π", "sigma": "σ",
        "lambda": "λ", "theta": "θ", "omega": "ω", "Omega": "Ω", "rightarrow": "→", "leftarrow": "←",
        "rightleftharpoons": "⇌", "Rightarrow": "⇒", "times": "×", "cdot": "·", "pm": "±", "leq": "≤",
        "geq": "≥", "approx": "≈", "neq": "≠", "infty": "∞", "degree": "°", "circ": "°"}

try:  # python-bidi changed its module layout between versions
    from bidi import get_display as _bidi
except Exception:
    try:
        from bidi.algorithm import get_display as _bidi
    except Exception:
        _bidi = None
try:
    import arabic_reshaper as _reshaper
except Exception:
    _reshaper = None


def has_ar(s: str) -> bool:
    return bool(AR_RE.search(s))


def is_rtl(s: str) -> bool:
    ar = len(AR_RE.findall(s))
    lat = len(re.findall(r"[A-Za-z]", s))
    return ar > 0 and ar >= lat


_NATIVE = {"ar": None}  # does this matplotlib already shape + reorder Arabic itself?


def shape(s: str) -> str:
    """Arabic needs letter joining + right-to-left reordering. Newer matplotlib does it natively
    (detected at start-up); otherwise we use arabic-reshaper + python-bidi."""
    if not has_ar(s) or "$" in s or _NATIVE["ar"]:
        return s
    try:
        if _reshaper:
            s = _reshaper.reshape(s)
        return _bidi(s) if _bidi else s
    except Exception:
        return s


def strip_math(s: str) -> str:
    s = s.replace("$", "")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:mathrm|mathbf|mathit|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _SYM.get(m.group(1), m.group(1)), s)
    s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"_(\w)", r"\1", s).replace("\\", " ")
    return re.sub(r"\s+", " ", s).strip()


def sanitize_math(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = (s.replace("\\text{", "\\mathrm{").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
          .replace("\\displaystyle", "").replace("\\boldsymbol", "\\mathbf").replace("\\limits", "")
          .replace("\\xrightarrow", "\\rightarrow").replace("\\longrightarrow", "\\rightarrow")
          .replace("\\cancel", ""))
    return re.sub(r"[\x00-\x08\x0b-\x1f]", "", s)


def fit_width(a: np.ndarray, max_w: int) -> np.ndarray:
    h, w = a.shape[:2]
    if w <= max_w or max_w <= 0:
        return a
    im = Image.fromarray(a).resize((int(max_w), max(1, int(h * max_w / w))), Image.LANCZOS)
    return np.array(im)


def tint(alpha: np.ndarray, color) -> np.ndarray:
    h, w = alpha.shape
    out = np.empty((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = alpha
    return out


class TextRenderer:
    """Draws text (and $math$) with matplotlib into alpha masks; wraps lines by pixel width."""
    DPI = 100

    def __init__(self, families):
        self.families = families
        rcParams["mathtext.default"] = "regular"
        rcParams["mathtext.fontset"] = "dejavusans"
        self._fig = Figure(figsize=(6, 1), dpi=self.DPI)
        self._r = FigureCanvasAgg(self._fig).get_renderer()
        self._cache = {}
        if _NATIVE["ar"] is None:
            _NATIVE["ar"] = True
            if _reshaper is not None:
                try:
                    raw = "\u0628" * 6
                    w_raw = self._bbox(raw, 60)[2]
                    w_res = self._bbox(_reshaper.reshape(raw), 60)[2]
                    _NATIVE["ar"] = abs(w_raw - w_res) < 0.12 * max(w_res, 1.0)
                except Exception:
                    _NATIVE["ar"] = False
            log("  Arabic shaping:", "native (matplotlib)" if _NATIVE["ar"] else "arabic-reshaper + bidi")

    def _fp(self, px):
        return fm.FontProperties(family=self.families, size=px * 72.0 / self.DPI)

    def _bbox(self, s, px):
        key = (s, int(px))
        if key not in self._cache:
            t = self._fig.text(0, 0, s, fontproperties=self._fp(px), va="baseline", ha="left")
            try:
                bb = t.get_window_extent(renderer=self._r)
                self._cache[key] = (bb.x0, bb.y0, bb.width, bb.height)
            finally:
                t.remove()
        return self._cache[key]

    def width(self, s, px) -> float:
        try:
            return self._bbox(s, px)[2]
        except Exception:
            return self._bbox(strip_math(s), px)[2]

    def wrap(self, s, px, max_w) -> list:
        lines, cur = [], ""
        for w in TOKEN_RE.findall(s):
            cand = f"{cur} {w}" if cur else w
            if cur and self.width(shape(cand), px) > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return lines or [""]

    def render(self, s, px) -> np.ndarray:
        try:
            return self._render(s, px)
        except Exception:
            return self._render(strip_math(s), px)

    def _render(self, s, px) -> np.ndarray:
        x0, y0, w, h = self._bbox(s, px)
        pad = max(4, int(px * 0.14))
        Wp, Hp = int(math.ceil(w)) + 2 * pad, int(math.ceil(h)) + 2 * pad
        fig = Figure(figsize=(Wp / self.DPI, Hp / self.DPI), dpi=self.DPI)
        fig.patch.set_alpha(0.0)
        canvas = FigureCanvasAgg(fig)
        fig.text((pad - x0) / Wp, (pad - y0) / Hp, s, fontproperties=self._fp(px), color="black",
                 va="baseline", ha="left")
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., 3].copy()


# ============================================================================ layout
@dataclass
class Part:
    rgba: np.ndarray
    dx: int = 0
    dy: int = 0
    direction: str = "ltr"  # ltr | rtl | ttb  (how it is revealed)


@dataclass
class Elem:
    parts: list
    w: int
    h: int
    kind: str
    chars: float = 10
    align: str = "left"  # left | center | right (inside the content area)
    gap: int = 14
    top: int = 0


class Ctx:
    def __init__(self, W, H, theme, families, sources=None):
        self.W, self.H = W, H
        self.s = H / 1080.0
        self.theme = THEMES.get(theme, THEMES["chalk"])
        self.tr = TextRenderer(families)
        self.left = int(110 * self.s)
        self.content_w = W - 2 * self.left
        self.title_y = int(44 * self.s)
        self.content_top = int(210 * self.s)
        self.bottom = H - int(72 * self.s)
        self.px = dict(text=int(58 * self.s), head=int(74 * self.s), math=int(66 * self.s), ar=int(56 * self.s),
                       title=int(88 * self.s), table=int(46 * self.s), flow=int(48 * self.s), cap=int(38 * self.s))
        self.bg = make_background(W, H, theme if theme in THEMES else "chalk")
        self.sources = sources or {}
        self._pens = {}

    def color(self, name):
        return self.theme["colors"].get(name or "white", self.theme["colors"]["white"])

    def pen(self, color):
        key = tuple(color)
        if key not in self._pens:
            self._pens[key] = pen_sprite(color, self.s)
        return self._pens[key]


def pen_sprite(color, s):
    S = int(130 * s)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tip = np.array([8.0 * s, S - 8.0 * s])
    ang = math.radians(52)
    u = np.array([math.cos(ang), -math.sin(ang)])
    n = np.array([-u[1], u[0]])

    def P(dist, half):
        v = tip + u * dist * s + n * half * s
        return (float(v[0]), float(v[1]))

    d.polygon([P(0, 0), P(18, -6), P(18, 6)], fill=tuple(color) + (255,))
    d.polygon([P(18, -9), P(100, -9), P(100, 9), P(18, 9)], fill=(236, 236, 236, 255), outline=(70, 70, 70, 255))
    d.polygon([P(72, -9.5), P(100, -9.5), P(100, 9.5), P(72, 9.5)], fill=tuple(color) + (255,))
    return np.array(img), (int(8 * s), int(S - 8 * s))


def layout_text(ctx, text, px, color, indent=0, prefix="", underline=False):
    tr = ctx.tr
    ind = int(indent * 64 * ctx.s)
    hang = int(tr.width(shape(prefix), px)) if prefix else 0
    max_w = ctx.content_w - ind
    pad = max(4, int(px * 0.14))
    pitch = int(px * 1.30)
    parts, y = [], 0
    for i, ln in enumerate(tr.wrap(prefix + text, px, max_w)):
        a = fit_width(tr.render(shape(ln), px), max_w)
        rtl = is_rtl(ln)
        x = ctx.content_w - a.shape[1] if rtl else ind + (hang if (i > 0 and prefix) else 0)
        parts.append(Part(tint(a, color), x, y, "rtl" if rtl else "ltr"))
        y += pitch
    if underline and parts:
        last = parts[-1]
        lw = max(3, int(4 * ctx.s))
        ul = np.zeros((lw, last.rgba.shape[1], 4), np.uint8)
        ul[..., :3], ul[..., 3] = color, 255
        parts.append(Part(ul, last.dx, last.dy + last.rgba.shape[0] - pad + int(2 * ctx.s), last.direction))
        y += lw + int(8 * ctx.s)
    return Elem(parts, ctx.content_w, y, "text", chars=len(text), gap=int(px * 0.20),
                top=int(px * 0.35) if underline else 0)


def layout_math(ctx, text, color):
    px = ctx.px["math"]
    s = sanitize_math(text)
    a = fit_width(ctx.tr.render(f"${s}$", px), ctx.content_w)
    return Elem([Part(tint(a, color))], a.shape[1], a.shape[0], "math", chars=len(text) * 1.2,
                align="center", gap=int(px * 0.22))


def _compose(w, h, layers):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for arr, x, y in layers:
        img.alpha_composite(Image.fromarray(arr), (max(0, int(x)), max(0, int(y))))
    return img


def layout_box(ctx, text, color):
    px, tr = ctx.px["text"], ctx.tr
    pad = int(26 * ctx.s)
    inner = ctx.content_w - 2 * pad - int(30 * ctx.s)
    arrs = [fit_width(tr.render(shape(l), px), inner) for l in tr.wrap(text, px, inner)]
    pitch = int(px * 1.25)
    w = max(a.shape[1] for a in arrs) + 2 * pad
    h = pitch * len(arrs) + 2 * pad - int(px * 0.1)
    lw = max(3, int(5 * ctx.s))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([lw // 2, lw // 2, w - 1 - lw // 2, h - 1 - lw // 2],
                                          radius=int(24 * ctx.s), outline=tuple(color) + (255,), width=lw)
    for i, a in enumerate(arrs):
        img.alpha_composite(Image.fromarray(tint(a, color)), (max(0, (w - a.shape[1]) // 2), pad - int(px * 0.05) + i * pitch))
    return Elem([Part(np.array(img), 0, 0, "ltr" if len(arrs) == 1 else "ttb")], w, h, "box",
                chars=len(text), align="center", gap=int(px * 0.3), top=int(px * 0.15))


def _arrow(d, p0, p1, color, lw, head):
    d.line([p0, p1], fill=tuple(color) + (255,), width=lw)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [p1] + [(p1[0] - head * math.cos(ang + k), p1[1] - head * math.sin(ang + k)) for k in (-0.5, 0.5)]
    d.polygon(pts, fill=tuple(color) + (255,))


def layout_flow(ctx, items, direction, color):
    px, tr, s = ctx.px["flow"], ctx.tr, ctx.s
    padx, pady, lw = int(24 * s), int(14 * s), max(3, int(4 * s))
    max_node = int(ctx.content_w * 0.30)
    nodes = []
    for it in items:
        arrs = [tr.render(shape(l), px) for l in tr.wrap(it, px, max_node - 2 * padx)]
        arrs = [fit_width(a, max_node - 2 * padx) for a in arrs]
        lh = int(px * 1.15)
        nodes.append((arrs, max(a.shape[1] for a in arrs) + 2 * padx, lh * len(arrs) + 2 * pady, lh))
    arrow = int(74 * s)
    horizontal = direction != "vertical" and sum(n[1] for n in nodes) + arrow * (len(nodes) - 1) <= ctx.content_w
    if horizontal:
        W = sum(n[1] for n in nodes) + arrow * (len(nodes) - 1)
        H = max(n[2] for n in nodes)
    else:
        arrow = int(56 * s)
        W = max(n[1] for n in nodes)
        H = sum(n[2] for n in nodes) + arrow * (len(nodes) - 1)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = y = 0
    for i, (arrs, nw, nh, lh) in enumerate(nodes):
        nx, ny = (x, (H - nh) // 2) if horizontal else ((W - nw) // 2, y)
        d.rounded_rectangle([nx, ny, nx + nw - 1, ny + nh - 1], radius=int(18 * s), outline=tuple(color) + (255,), width=lw)
        for j, a in enumerate(arrs):
            img.alpha_composite(Image.fromarray(tint(a, color)), (nx + (nw - a.shape[1]) // 2, ny + pady + j * lh))
        if i < len(nodes) - 1:
            if horizontal:
                _arrow(d, (nx + nw + 8 * s, H // 2), (nx + nw + arrow - 8 * s, H // 2), color, lw, 18 * s)
                x += nw + arrow
            else:
                _arrow(d, (W // 2, ny + nh + 6 * s), (W // 2, ny + nh + arrow - 6 * s), color, lw, 18 * s)
                y += nh + arrow
    return Elem([Part(np.array(img), 0, 0, "ltr" if horizontal else "ttb")], W, H, "flow", chars=len(items),
                align="center", gap=int(24 * s), top=int(8 * s))


def layout_table(ctx, rows, color):
    px, tr, s = ctx.px["table"], ctx.tr, ctx.s
    cells = [[c.strip() for c in r.split("|")] for r in rows]
    ncol = max(len(r) for r in cells)
    cells = [r + [""] * (ncol - len(r)) for r in cells]
    pad = int(14 * s)
    nat = [int(math.ceil(max(tr.width(shape(r[j]), px) for r in cells))) + 2 * pad for j in range(ncol)]
    total = sum(nat)
    colw = nat if total <= ctx.content_w else [int(n * ctx.content_w / total) for n in nat]
    pitch = int(px * 1.2)
    wrapped = [[tr.wrap(c, px, colw[j] - 2 * pad) for j, c in enumerate(r)] for r in cells]
    rowh = [max(len(c) for c in r) * pitch + 2 * pad for r in wrapped]
    W, H = int(sum(colw)), int(sum(rowh))
    lw = max(2, int(3 * s))
    img = Image.new("RGBA", (W + lw, H + lw), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    hdr = ctx.color("yellow")
    y = 0
    for i, r in enumerate(wrapped):
        x = 0
        for j, lines in enumerate(r):
            d.rectangle([x, y, x + colw[j], y + rowh[i]], outline=tuple(color) + (200,), width=lw)
            for k, ln in enumerate(lines):
                a = fit_width(tr.render(shape(ln), px), colw[j] - 2 * pad)
                ax = x + colw[j] - pad - a.shape[1] if is_rtl(ln) else x + pad
                img.alpha_composite(Image.fromarray(tint(a, hdr if i == 0 else color)), (int(ax), int(y + pad + k * pitch - px * 0.05)))
            x += colw[j]
        if i == 0:
            d.line([(0, y + rowh[0]), (W, y + rowh[0])], fill=tuple(hdr) + (255,), width=lw + 2)
        y += rowh[i]
    return Elem([Part(np.array(img), 0, 0, "ttb")], W, H, "table", chars=len(rows), align="center",
                gap=int(24 * s), top=int(10 * s))


def layout_divider(ctx, color):
    h = int(18 * ctx.s)
    a = np.zeros((h, ctx.content_w, 4), np.uint8)
    a[h // 2 - 1:h // 2 + 2, :, :3] = color
    a[h // 2 - 1:h // 2 + 2, :, 3] = 150
    return Elem([Part(a)], ctx.content_w, h, "divider", chars=4, gap=int(10 * ctx.s))


def layout_image(ctx, item, avail_h):
    path = ctx.sources.get(item.get("file"))
    if not path:
        return None
    if str(path).lower().endswith(".pdf"):
        import fitz  # PyMuPDF; ImportError -> item skipped by caller

        doc = fitz.open(str(path))
        pg = min(max(int(item.get("page", 1)), 1), len(doc)) - 1
        pix = doc[pg].get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(path).convert("RGB")
    crop = item.get("crop")
    if crop:
        y0, x0, y1, x1 = crop
        if y1 > y0 + 20 and x1 > x0 + 20:
            W0, H0 = img.size
            img = img.crop((int(x0 / 1000 * W0), int(y0 / 1000 * H0), int(x1 / 1000 * W0), int(y1 / 1000 * H0)))
    cap = item.get("caption") or ""
    cap_h = int(ctx.px["cap"] * 1.5) if cap else 0
    max_w, max_h = int(ctx.content_w * 0.92), max(120, avail_h - cap_h - int(20 * ctx.s))
    sc = min(max_w / img.width, max_h / img.height, 1.6)
    img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.LANCZOS)
    b = max(3, int(5 * ctx.s))
    canvas = Image.new("RGBA", (img.width + 2 * b, img.height + 2 * b + cap_h), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).rectangle([0, 0, img.width + 2 * b - 1, img.height + 2 * b - 1], fill=(255, 255, 255, 255))
    canvas.paste(img.convert("RGBA"), (b, b))
    if cap:
        a = fit_width(ctx.tr.render(shape(cap), ctx.px["cap"]), canvas.width)
        canvas.alpha_composite(Image.fromarray(tint(a, ctx.color("orange"))), (0, img.height + 2 * b + int(4 * ctx.s)))
    return Elem([Part(np.array(canvas), 0, 0, "ttb")], canvas.width, canvas.height, "image", chars=10,
                align="center", gap=int(20 * ctx.s))


def layout_title(ctx, text):
    px, color = ctx.px["title"], ctx.color("yellow")
    a = fit_width(ctx.tr.render(shape(text), px), ctx.content_w)
    h, w = a.shape
    lw = max(3, int(5 * ctx.s))
    pad = max(4, int(px * 0.14))
    img = np.zeros((h + lw + int(10 * ctx.s), w, 4), np.uint8)
    img[:h] = tint(a, color)
    uy = h - pad + int(4 * ctx.s)
    img[uy:uy + lw, :, :3], img[uy:uy + lw, :, 3] = color, 255
    return Elem([Part(img)], w, img.shape[0], "title", chars=len(text))


def layout_item(ctx, it, avail_h):
    t, c = it["type"], it.get("color")
    try:
        if t == "heading":
            return layout_text(ctx, it["text"], ctx.px["head"], ctx.color(c or "yellow"), underline=True)
        if t == "text":
            return layout_text(ctx, it["text"], ctx.px["text"], ctx.color(c), indent=it.get("indent", 0))
        if t == "bullet":
            return layout_text(ctx, it["text"], ctx.px["text"], ctx.color(c), indent=it.get("indent", 0), prefix="•  ")
        if t == "ar":
            return layout_text(ctx, it["text"], ctx.px["ar"], ctx.color(c or "green"))
        if t == "math":
            return layout_math(ctx, it["text"], ctx.color(c or "green"))
        if t == "box":
            return layout_box(ctx, it["text"], ctx.color(c or "yellow"))
        if t == "flow":
            return layout_flow(ctx, it["items"], it.get("direction", "horizontal"), ctx.color(c or "blue"))
        if t == "table":
            return layout_table(ctx, it["rows"], ctx.color(c))
        if t == "divider":
            return layout_divider(ctx, ctx.color(c))
        if t == "page_image":
            return layout_image(ctx, it, avail_h)
    except Exception as e:
        log(f"    layout of '{t}' failed, item skipped: {type(e).__name__}: {str(e)[:120]}")
    return None


# ============================================================================ planning
CLEAR_T, TITLE_T, END_HOLD = 0.6, 0.9, 1.2


@dataclass
class Event:
    kind: str  # clear | write | pause
    t0: float
    t1: float
    elem: Elem | None = None
    x: int = 0
    y: int = 0


def nominal(e: Elem) -> float:
    k = e.kind
    if k == "text":
        return max(0.8, min(9.0, e.chars / 15.0))
    if k == "math":
        return max(1.4, min(6.0, 1.0 + e.chars * 0.10))
    if k == "box":
        return max(1.2, min(7.0, 0.5 + e.chars / 14.0))
    if k == "flow":
        return 1.0 + 0.5 * e.chars
    if k == "table":
        return 1.0 + 0.7 * e.chars
    if k == "image":
        return 1.2
    return 0.6


def plan(steps, audios, ctx):
    """Returns (events, step_len, chapters, total_seconds)."""
    events, step_len, chapters = [], [], []
    st_ = {"y": ctx.content_top, "blank": True, "title": "", "section": -1}
    t = 0.0

    def page_actions(title):
        acts = []
        if not st_["blank"]:
            acts.append(("clear", None, 0, 0, CLEAR_T))
        acts.append(("title", layout_title(ctx, title), ctx.left, ctx.title_y, TITLE_T))
        st_["y"], st_["blank"] = ctx.content_top, False
        return acts

    for step, au in zip(steps, audios):
        s0, A = t, au.total
        if step.section != st_["section"]:
            chapters.append((s0, step.section_title))
            st_["section"] = step.section
        acts = []
        if step.new_board or st_["blank"]:
            st_["title"] = step.board_title or st_["title"] or step.section_title
            acts += page_actions(st_["title"])
        for it in step.board:
            e = layout_item(ctx, it, ctx.bottom - st_["y"])
            if e is None:
                continue
            if st_["y"] > ctx.content_top:
                st_["y"] += e.top
            if st_["y"] + e.h > ctx.bottom and st_["y"] > ctx.content_top:  # board full -> new page
                base = st_["title"].replace(" (cont.)", "")
                st_["title"] = base + " (cont.)"
                acts += page_actions(st_["title"])
                if it["type"] == "page_image":
                    e = layout_item(ctx, it, ctx.bottom - st_["y"]) or e
            x = ctx.left + {"left": 0, "center": (ctx.content_w - e.w) // 2, "right": ctx.content_w - e.w}[e.align]
            acts.append(("write", e, x, st_["y"], nominal(e)))
            st_["y"] += e.h + e.gap
        fixed = sum(a[4] for a in acts if a[0] != "write")
        writes = sum(a[4] for a in acts if a[0] == "write")
        window = au.speech_end - 0.15 - fixed
        factor = min(writes / window, 2.5) if writes > 0 and window > 0.5 and writes > window else 1.0
        lt = 0.15
        for kind, e, x, y, nom in acts:
            dur = nom / factor if kind == "write" else nom
            events.append(Event("clear" if kind == "clear" else "write", s0 + lt, s0 + lt + dur, e, x, y))
            lt += dur
        L = max(A, lt + 0.35)
        if step.pause > 0:
            events.append(Event("pause", s0 + L - step.pause, s0 + L))
        step_len.append(L)
        t += L
    return events, step_len, chapters, t + END_HOLD


def write_wav(audios, step_len, total, path: Path):
    chunks, cum = [], 0.0
    for au, L in zip(audios, step_len):
        n = int(round((cum + L) * SR)) - int(round(cum * SR))
        s = au.samples[:n]
        chunks.append(np.concatenate([s, np.zeros(n - len(s), np.int16)]) if len(s) < n else s)
        cum += L
    rest = int(round(total * SR)) - sum(len(c) for c in chunks)
    if rest > 0:
        chunks.append(np.zeros(rest, np.int16))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.concatenate(chunks).tobytes())


# ============================================================================ rendering
def blit(frame, img, x, y, direction="ltr", frac=1.0):
    """Alpha-blend img (RGBA) onto frame (RGB) at x,y, revealing only `frac` of it. Returns pen position."""
    h, w = img.shape[:2]
    if direction == "ttb":
        ch = max(1, int(round(h * frac)))
        sub, x0, y0 = img[:ch], x, y
        pen = (x0 + int(0.04 * w), y0 + ch)
    elif direction == "rtl":
        cw = max(1, int(round(w * frac)))
        sub, x0, y0 = img[:, w - cw:], x + w - cw, y
        pen = (x0, y0 + h // 2)
    else:
        cw = max(1, int(round(w * frac)))
        sub, x0, y0 = img[:, :cw], x, y
        pen = (x0 + cw, y0 + h // 2)
    sh, sw = sub.shape[:2]
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + sw), min(frame.shape[0], y0 + sh)
    if fx1 > fx0 and fy1 > fy0:
        sub = sub[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        a = sub[..., 3:4].astype(np.float32) * (1 / 255.0)
        reg = frame[fy0:fy1, fx0:fx1]
        reg[:] = (reg * (1.0 - a) + sub[..., :3] * a + 0.5).astype(np.uint8)
    return pen


def draw_elem(frame, e: Elem, x, y, p):
    lens = [pt.rgba.shape[0] if pt.direction == "ttb" else pt.rgba.shape[1] for pt in e.parts]
    total = float(sum(lens)) or 1.0
    target, acc, pen = p * total, 0.0, None
    for pt, ln in zip(e.parts, lens):
        frac = 1.0 if acc + ln <= target + 1e-6 else (target - acc) / ln
        if frac <= 0:
            break
        pen = blit(frame, pt.rgba, x + pt.dx, y + pt.dy, pt.direction, min(1.0, frac))
        acc += ln
    return pen


def countdown_sprite(frac, s):
    S = int(130 * s)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box = [6, 6, S - 7, S - 7]
    d.ellipse(box, fill=(0, 0, 0, 90), outline=(255, 255, 255, 230), width=max(3, int(5 * s)))
    if frac > 0.01:
        d.pieslice([14, 14, S - 15, S - 15], -90, -90 + 360 * frac, fill=(255, 227, 112, 235))
    return np.array(img)


def render_video(events, total, ctx: Ctx, wav: Path, out: Path, fps=15):
    W, H = ctx.W, ctx.H
    preset, crf = env("X264_PRESET", "veryfast"), env("X264_CRF", "23")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-framerate", str(fps), "-i", "pipe:0", "-i", str(wav), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", str(fps * 2),
           "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-movflags", "+faststart", str(out)]
    WORK.mkdir(parents=True, exist_ok=True)
    errlog = open(WORK / "ffmpeg.log", "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)
    base = ctx.bg.copy()
    state = {"f": 0}

    def emit(data: bytes, n=1):
        for _ in range(n):
            proc.stdin.write(data)
        state["f"] += n

    def idle_until(t):
        n = int(round(t * fps)) - state["f"]
        if n > 0:
            emit(base.tobytes(), n)

    try:
        for ev in events:
            idle_until(ev.t0)
            n = max(1, int(round(ev.t1 * fps)) - state["f"])
            if ev.kind == "clear":
                bg = ctx.bg.astype(np.float32)
                b32 = base.astype(np.float32)
                for k in range(1, n + 1):
                    a = k / n
                    emit((b32 * (1 - a) + bg * a).astype(np.uint8).tobytes())
                base = ctx.bg.copy()
            elif ev.kind == "write":
                color = tuple(int(v) for v in ctx.theme["colors"]["white"])
                # pen colour = colour of the first visible pixel row of the element
                for pt in ev.elem.parts:
                    nz = np.argwhere(pt.rgba[..., 3] > 200)
                    if len(nz):
                        color = tuple(int(v) for v in pt.rgba[nz[0][0], nz[0][1], :3])
                        break
                sprite, (tx, ty) = ctx.pen(color)
                for k in range(1, n + 1):
                    fr = base.copy()
                    pen = draw_elem(fr, ev.elem, ev.x, ev.y, k / n)
                    if pen and k < n:
                        blit(fr, sprite, pen[0] - tx, pen[1] - ty)
                    emit(fr.tobytes())
                draw_elem(base, ev.elem, ev.x, ev.y, 1.0)
            elif ev.kind == "pause":
                for k in range(1, n + 1):
                    fr = base.copy()
                    spr = countdown_sprite(1 - k / n, ctx.s)
                    blit(fr, spr, W - int(210 * ctx.s), H - int(200 * ctx.s))
                    emit(fr.tobytes())
        idle_until(total)
        proc.stdin.close()
        rc = proc.wait()
    except BrokenPipeError:
        proc.wait()
        raise RuntimeError("ffmpeg stopped early:\n" + (WORK / "ffmpeg.log").read_text(errors="ignore")[-1500:])
    if rc != 0:
        raise RuntimeError("ffmpeg failed:\n" + (WORK / "ffmpeg.log").read_text(errors="ignore")[-1500:])
    errlog.close()
