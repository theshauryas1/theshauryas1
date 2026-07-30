#!/usr/bin/env python3
"""
generate_banner.py
==================
Generates dark.svg and light.svg for the theshauryas1 GitHub profile.

Pipeline (per the master prompt spec):
  1. Load & pre-process photo (autocontrast + UnsharpMask, contrast ×1.3)
  2. Segment background via colour-distance threshold + morphological closing
     -> dark mode: only subject dots on dark panel (background cleared)
     -> light mode: full photo dithering
  3. Floyd–Steinberg dither, serpentine scan, 300×340 grid, 1-bit output
  4. Build SVG <path> runs, shape-rendering="crispEdges", no font glyphs
  5. Scatter dots into ~60 intro animation groups (evenness sigma < 0.05)
  6. ~94 drift bands for portrait->logo dissolve
  7. ~900 traveller dots morphed via optimal transport between 6 tech logos
  8. Assemble full SVG with info panel, LIVE badge, social links

Usage:
  pip install -r requirements.txt
  python generate_banner.py --photo path/to/photo.jpg [--out-dir .]

Output:
  dark.svg   (~900KB)
  light.svg  (~900KB)
"""

import argparse
import math
import random
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# --- Optional rembg (background removal) ------------------------------------
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    print("[WARN] rembg not available — using colour-distance fallback for BG removal")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

CFG = dict(
    # SVG canvas
    SVG_W=1180, SVG_H=610,
    # Portrait grid — 150x170 keeps 15:17 aspect ratio, optimizes SVG file size < 1.5MB for GitHub Proxy
    GRID_W=150, GRID_H=170,
    DOT_R=1.35,              # dot radius (px) adjusted for grid pitch
    DOT_PITCH=1,             # 1 dot per cell
    # Contrast / sharpening
    CONTRAST=1.3,
    UNSHARP_RADIUS=3,
    UNSHARP_PCT=140,
    AUTOCONTRAST_CUTOFF=1,
    # Background segmentation (colour-distance fallback)
    BG_DIST_THRESH=30,
    BG_CLOSE_R=5,            # morphological closing radius
    # Animation timing (seconds)
    INTRO_DUR=3.2,
    INTRO_FADE_START=0.0,
    INTRO_FADE_END=2.0,
    LOOP_DUR=14.2,
    PORTRAIT_HOLD=3.0,
    LOGO_HOLD=2.0,
    LOGO_TRANS=1.3,
    # Portrait drift
    N_DRIFT_BANDS=60,
    DRIFT_NOISE_SIGMA=4,
    DRIFT_FRACTION=0.42,
    # Traveller dots
    N_TRAVELLERS=300,
    # Colour palette
    BG_DARK='#0A101F',
    PANEL_DARK='#101826',
    PORTRAIT_HUE='#A78BFA',   # violet — distinct from cyan UI chrome
    CHROME_DARK='#22D3EE',
    CHROME_LIGHT='#0891B2',
    ACCENT='#10B981',
    DANGER='#EF4444',
    TEXT_DARK='#F8FAFC',
    MUTED_DARK='#94A3B8',
    BG_LIGHT='#F0F7FF',
    PANEL_LIGHT='#FFFFFF',
    PORTRAIT_HUE_LIGHT='#7C3AED',
    TEXT_LIGHT='#1E293B',
    MUTED_LIGHT='#64748B',
    # Empty-cell dot colour for dark snake (must be visible)
    SNAKE_EMPTY_DARK='#2d3343',
)

# ----------------------------------------------------------------------------
# LOGO PATH DATA  (SVG path strings, viewBox 0 0 100 100)
# ----------------------------------------------------------------------------

LOGOS = {
    "PyTorch": dict(
        label="PyTorch",
        sub="Deep Learning",
        color="#EE4C2C",
        # Stylised flame + circle
        paths=[
            "M50 8 C60 20 72 38 60 55 C52 67 38 67 32 55 C22 38 42 24 48 12 C44 22 36 30 40 42 C44 54 56 54 58 44 C62 30 52 18 50 8Z",
            "M62 58 A8 8 0 1 1 46 58 A8 8 0 1 1 62 58Z",
        ],
    ),
    "NVIDIA": dict(
        label="NVIDIA",
        sub="GPU Computing",
        color="#76B900",
        paths=[
            # Eye shape
            "M15 50 Q30 25 50 22 Q70 20 85 50 Q70 78 50 78 Q30 75 15 50Z",
            # Iris
            "M38 50 A12 12 0 1 1 62 50 A12 12 0 1 1 38 50Z",
        ],
    ),
    "Kubernetes": dict(
        label="Kubernetes",
        sub="Orchestration",
        color="#326CE5",
        paths=[
            # Outer ring
            "M50 8 A42 42 0 1 1 49.9 8Z M50 16 A34 34 0 1 0 50.1 16Z",
            # 8 spokes
            "M50 16 L50 36 M50 64 L50 84 M16 50 L36 50 M64 50 L84 50 M26 26 L40 40 M60 60 L74 74 M74 26 L60 40 M40 60 L26 74",
            # Hub
            "M38 50 A12 12 0 1 1 62 50 A12 12 0 1 1 38 50Z",
        ],
    ),
    "FastAPI": dict(
        label="FastAPI",
        sub="Backend API",
        color="#009688",
        paths=[
            # Lightning bolt
            "M58 8 L28 52 L48 52 L42 92 L72 48 L52 48Z",
        ],
    ),
    "LangChain": dict(
        label="LangChain",
        sub="LLM Framework",
        color="#22D3EE",
        paths=[
            # Three chain links
            "M10 42 A14 14 0 0 1 10 58 A14 14 0 0 1 10 42Z M10 42 L36 42 A14 14 0 0 1 36 58 L10 58",
            "M36 42 A14 14 0 0 1 36 58 L64 58 A14 14 0 0 1 64 42 L36 42Z",
            "M64 42 A14 14 0 0 1 64 58 L90 58 A14 14 0 0 1 90 42 L64 42Z",
        ],
    ),
    "Docker": dict(
        label="Docker",
        sub="Containers",
        color="#2496ED",
        paths=[
            # Whale body
            "M20 55 Q50 38 80 55 Q70 75 50 75 Q30 75 20 55Z",
            # Container stack
            "M22 42 L38 42 L38 52 L22 52Z M42 42 L58 42 L58 52 L42 52Z M62 42 L78 42 L78 52 L62 52Z",
            "M22 28 L38 28 L38 38 L22 38Z M42 28 L58 28 L58 38 L42 38Z",
        ],
    ),
}

LOGO_ORDER = ["PyTorch", "NVIDIA", "Kubernetes", "FastAPI", "LangChain", "Docker"]


# ----------------------------------------------------------------------------
# SYSTEM.INFO ROWS
# ----------------------------------------------------------------------------

INFO_ROWS = [
    ("Subject",    "Shaurya Sharma"),
    ("Role",       "AI/ML Engineer"),
    ("Origin",     "India 🇮🇳"),
    ("Education",  "Automation & Robotics"),
    ("Status",     "Building Agentic AI"),
    (None, None),  # divider
    ("Core.Lang",  "Python • C++ • SQL"),
    ("Core.AI",    "PyTorch • TensorFlow"),
    ("Core.Agents","LangGraph • CrewAI • AutoGen"),
    ("Core.Back",  "FastAPI • gRPC"),
    ("Core.DB",    "PostgreSQL • MongoDB • FAISS"),
    ("Core.Infra", "Docker • K8s • Terraform"),
    ("Core.Cloud", "AWS • Azure • GCP"),
    (None, None),
    ("Grid.Mail",  "—"),
    ("Grid.Port",  "shaurya-beta.vercel.app"),
    ("Grid.LinkedIn","theshauryas1"),
    ("Grid.GitHub","theshauryas1"),
]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def preprocess(img: Image.Image) -> Image.Image:
    """Crop head+shoulders, apply contrast ×1.3, autocontrast, UnsharpMask."""
    w, h = img.size
    # Head+shoulders crop: top 75%, centred horizontally
    crop_h = int(h * 0.78)
    crop_w = min(w, int(crop_h * (300 / 340)))
    x0 = (w - crop_w) // 2
    img = img.crop((x0, 0, x0 + crop_w, crop_h))
    img = img.resize((CFG['GRID_W'], CFG['GRID_H']), Image.LANCZOS)
    img = img.convert("RGB")
    # Autocontrast
    img = ImageOps.autocontrast(img, cutoff=CFG['AUTOCONTRAST_CUTOFF'])
    # Contrast ×1.3
    img = ImageEnhance.Contrast(img).enhance(CFG['CONTRAST'])
    # UnsharpMask
    img = img.filter(ImageFilter.UnsharpMask(
        radius=CFG['UNSHARP_RADIUS'],
        percent=CFG['UNSHARP_PCT'],
        threshold=3
    ))
    return img


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — BACKGROUND SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════

def segment_background_rembg(img_rgb: Image.Image) -> np.ndarray:
    """Use rembg neural net -> alpha mask (255 = subject, 0 = background)."""
    result = rembg_remove(img_rgb)
    alpha = np.array(result)[:, :, 3]
    return (alpha > 128).astype(np.uint8)


def segment_background_colordist(img_rgb: Image.Image) -> np.ndarray:
    """
    Colour-distance fallback:
    Sample corners for background colour, threshold by Euclidean distance,
    binary closing, flood-fill holes, keep largest connected component.
    """
    from scipy import ndimage

    arr = np.array(img_rgb, dtype=np.float32)
    H, W = arr.shape[:2]
    # Sample background from corners (5×5 patches)
    r = 5
    corners = np.concatenate([
        arr[:r, :r].reshape(-1, 3),
        arr[:r, -r:].reshape(-1, 3),
        arr[-r:, :r].reshape(-1, 3),
        arr[-r:, -r:].reshape(-1, 3),
    ])
    bg_color = corners.mean(axis=0)
    # Colour distance
    dist = np.linalg.norm(arr - bg_color, axis=2)
    fg_mask = (dist > CFG['BG_DIST_THRESH']).astype(np.uint8)
    # Morphological closing to fill small holes
    struct = ndimage.generate_binary_structure(2, 1)
    for _ in range(CFG['BG_CLOSE_R']):
        fg_mask = ndimage.binary_dilation(fg_mask, struct).astype(np.uint8)
    for _ in range(CFG['BG_CLOSE_R']):
        fg_mask = ndimage.binary_erosion(fg_mask, struct, border_value=1).astype(np.uint8)
    # Fill holes
    filled = ndimage.binary_fill_holes(fg_mask)
    # Keep largest component
    labeled, n = ndimage.label(filled)
    if n == 0:
        return fg_mask
    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest).astype(np.uint8)


def get_fg_mask(img_rgb: Image.Image) -> np.ndarray:
    if HAS_REMBG:
        try:
            return segment_background_rembg(img_rgb)
        except Exception as e:
            print(f"[WARN] rembg failed ({e}), falling back to colour-distance")
    return segment_background_colordist(img_rgb)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — FLOYD–STEINBERG DITHER (serpentine)
# ════════════════════════════════════════════════════════════════════════════

def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """
    1-bit Floyd–Steinberg dither, serpentine scan.
    Input: H×W float32 in [0,1]
    Output: H×W uint8 {0,1}  (1 = ink dot)
    """
    buf = gray.astype(np.float32).copy()
    H, W = buf.shape
    out = np.zeros_like(buf, dtype=np.uint8)

    for y in range(H):
        if y % 2 == 0:
            xs = range(W)
            fw = +1
        else:
            xs = range(W - 1, -1, -1)
            fw = -1

        for x in xs:
            old = buf[y, x]
            new = 1.0 if old > 0.5 else 0.0
            out[y, x] = int(new)
            err = old - new
            # Diffuse error: right, lower-left, lower, lower-right
            # Serpentine: directions flip with scan direction
            nx = x + fw
            if 0 <= nx < W:
                buf[y, nx] = np.clip(buf[y, nx] + err * 7 / 16, 0, 1)
            if y + 1 < H:
                px = x - fw
                if 0 <= px < W:
                    buf[y + 1, px] = np.clip(buf[y + 1, px] + err * 3 / 16, 0, 1)
                buf[y + 1, x] = np.clip(buf[y + 1, x] + err * 5 / 16, 0, 1)
                nx2 = x + fw
                if 0 <= nx2 < W:
                    buf[y + 1, nx2] = np.clip(buf[y + 1, nx2] + err * 1 / 16, 0, 1)

    return out


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — SVG PATH RUNS
# ════════════════════════════════════════════════════════════════════════════

def dot_grid_to_paths(
    dot_mask: np.ndarray,
    ox: float, oy: float,
    dot_r: float,
    color: str,
    extra_attrs: str = "",
) -> list[str]:
    """
    Convert 1-bit grid to SVG <path> commands (horizontal runs).
    Returns list of <path> element strings.
    shape-rendering="crispEdges" on the group, not per-dot.
    """
    H, W = dot_mask.shape
    paths = []
    r = dot_r

    for y in range(H):
        row = dot_mask[y]
        x = 0
        while x < W:
            if row[x] == 1:
                # find run end
                x_start = x
                while x < W and row[x] == 1:
                    x += 1
                x_end = x
                # emit single rect-like path for this run
                # (individual dots are circles — but we use rects for performance)
                cx = ox + x_start + r
                cy_top = oy + y + 0.5 - r
                run_w = (x_end - x_start) * 1 - (1 - 2 * r)
                run_h = 2 * r
                p = (
                    f'<rect x="{cx - r:.2f}" y="{cy_top:.2f}" '
                    f'width="{run_w:.2f}" height="{run_h:.2f}"'
                )
                if extra_attrs:
                    p += f" {extra_attrs}"
                p += "/>"
                paths.append(p)
            else:
                x += 1
    return paths


def collect_dot_positions(dot_mask: np.ndarray, ox: float, oy: float) -> list[tuple]:
    """Return list of (cx, cy) for all ink dots in the mask."""
    H, W = dot_mask.shape
    positions = []
    for y in range(H):
        for x in range(W):
            if dot_mask[y, x] == 1:
                positions.append((ox + x + 0.5, oy + y + 0.5))
    return positions


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — INTRO ANIMATION GROUPS (scattered, evenness < 0.05)
# ════════════════════════════════════════════════════════════════════════════

def make_intro_groups(positions: list, n_groups: int = 60) -> list[list[int]]:
    """
    Assign each dot to a group such that each group is spatially scattered
    (not a contiguous region). Uses round-robin on a shuffled index.
    Verify evenness with check_intro_evenness().
    """
    idx = list(range(len(positions)))
    random.shuffle(idx)
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for i, dot_idx in enumerate(idx):
        groups[i % n_groups].append(dot_idx)
    return groups


def check_intro_evenness(positions: list, groups: list[list[int]]) -> float:
    """
    Evenness metric: std of per-group centroid distances from overall centroid.
    < 0.05 = well scattered, > 0.2 = patchy.
    """
    if not positions:
        return 0.0
    all_x = np.array([p[0] for p in positions])
    all_y = np.array([p[1] for p in positions])
    cx, cy = all_x.mean(), all_y.mean()
    W = max(all_x) - min(all_x) + 1
    H = max(all_y) - min(all_y) + 1
    dists = []
    for g in groups:
        if not g:
            continue
        gx = np.mean([positions[i][0] for i in g])
        gy = np.mean([positions[i][1] for i in g])
        d = math.sqrt(((gx - cx) / W) ** 2 + ((gy - cy) / H) ** 2)
        dists.append(d)
    return float(np.std(dists)) if dists else 0.0


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — DRIFT BANDS (portrait -> logo dissolve)
# ════════════════════════════════════════════════════════════════════════════

def make_drift_bands(
    positions: list,
    n_bands: int,
    logo_centroid: tuple,
    drift_fraction: float,
    noise_sigma: float,
) -> list[list[int]]:
    """
    Group dots into drift bands. Each band has a unique translate offset.
    Per-dot noise (sigma=noise_sigma) is added BEFORE grouping to prevent
    the mathematical grid trap (boundary metric < 0.01 is organic).
    """
    arr = np.array(positions, dtype=np.float32)
    cx, cy = logo_centroid
    # Direction to centroid
    dx = arr[:, 0] - cx
    dy = arr[:, 1] - cy
    dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    # Normalised projection + noise
    proj = dist + np.random.normal(0, noise_sigma, size=len(positions))
    # Assign to bands by quantile
    bands: list[list[int]] = [[] for _ in range(n_bands)]
    order = np.argsort(proj)
    for rank, idx in enumerate(order):
        bands[int(rank * n_bands / len(order))].append(int(idx))
    return bands


def check_straight_boundary(positions: list, bands: list[list[int]]) -> float:
    """
    Straight-boundary metric: ~0.01 = organic, ~0.17 = grid artefact.
    Measures linearity of band boundaries via variance of y values at
    transitions between adjacent bands.
    """
    if len(positions) < 2:
        return 0.0
    label = np.zeros(len(positions), dtype=int)
    for i, band in enumerate(bands):
        for idx in band:
            label[idx] = i
    arr = np.array(positions)
    xs = arr[:, 0]
    ys = arr[:, 1]
    # For each unique x, measure variation in band label across y
    unique_x = np.unique(xs.astype(int))
    variances = []
    for ux in unique_x[:50]:  # sample
        mask = (xs.astype(int) == ux)
        labels_at_x = label[mask]
        if len(labels_at_x) > 1:
            variances.append(float(np.std(labels_at_x)))
    return float(np.mean(variances)) if variances else 0.0


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — OPTIMAL TRANSPORT (traveller dots)
# ════════════════════════════════════════════════════════════════════════════

def sample_logo_dots(logo_name: str, n: int, ox: float, oy: float,
                     panel_w: float, panel_h: float) -> np.ndarray:
    """
    Sample n points from a logo's filled path(s).
    Uses rejection sampling inside a bounding grid.
    Returns (n, 2) array of (x, y) positions.
    """
    from scipy.spatial import ConvexHull  # noqa

    logo = LOGOS[logo_name]
    # Build a rasterised bitmap of the logo in panel coordinates
    # (100×100 -> panel_w × panel_h)
    scale_x = panel_w / 100.0
    scale_y = panel_h / 100.0
    # Rasterise using a simple polygon fill (scanline)
    bitmap = np.zeros((int(panel_h), int(panel_w)), dtype=np.uint8)

    # Draw filled path approximations via scanline
    _draw_logo_bitmap(bitmap, logo, scale_x, scale_y)

    ys, xs = np.where(bitmap > 0)
    if len(xs) == 0:
        # fallback: random within panel
        pts = np.random.rand(n, 2)
        pts[:, 0] = pts[:, 0] * panel_w + ox
        pts[:, 1] = pts[:, 1] * panel_h + oy
        return pts

    # Sample n points from filled pixels
    indices = np.random.choice(len(xs), size=n, replace=len(xs) < n)
    pts = np.column_stack([xs[indices].astype(float), ys[indices].astype(float)])
    # Add sub-pixel jitter
    pts += np.random.uniform(-0.5, 0.5, pts.shape)
    pts[:, 0] += ox
    pts[:, 1] += oy
    return pts


def _draw_logo_bitmap(bitmap: np.ndarray, logo: dict, sx: float, sy: float):
    """Simple scanline fill of logo path approximations."""
    H, W = bitmap.shape
    # For each logo, use a simplified polygon from the path data
    # We'll draw a coarse approximation using polylines
    name = logo["label"]

    if name == "PyTorch":
        # Flame shape
        _fill_poly(bitmap, [
            (50 * sx, 8 * sy), (60 * sx, 20 * sy), (72 * sx, 38 * sy),
            (60 * sx, 55 * sy), (52 * sx, 67 * sy), (38 * sx, 67 * sy),
            (32 * sx, 55 * sy), (22 * sx, 38 * sy), (42 * sx, 24 * sy),
        ])
        _fill_circle(bitmap, int(62 * sx), int(58 * sy), int(8 * min(sx, sy)))

    elif name == "NVIDIA":
        _fill_ellipse(bitmap, int(50 * sx), int(50 * sy), int(35 * sx), int(28 * sy))
        # Punch out inner background
        _fill_circle(bitmap, int(50 * sx), int(50 * sy), int(14 * min(sx, sy)), val=0)

    elif name == "Kubernetes":
        _fill_circle(bitmap, int(50 * sx), int(50 * sy), int(42 * min(sx, sy)))
        _fill_circle(bitmap, int(50 * sx), int(50 * sy), int(30 * min(sx, sy)), val=0)
        # Add hub
        _fill_circle(bitmap, int(50 * sx), int(50 * sy), int(12 * min(sx, sy)))
        # Spokes (3px wide)
        _draw_line_thick(bitmap, int(50 * sx), int(16 * sy), int(50 * sx), int(36 * sy), 3)
        _draw_line_thick(bitmap, int(50 * sx), int(64 * sy), int(50 * sx), int(84 * sy), 3)
        _draw_line_thick(bitmap, int(16 * sx), int(50 * sy), int(36 * sx), int(50 * sy), 3)
        _draw_line_thick(bitmap, int(64 * sx), int(50 * sy), int(84 * sx), int(50 * sy), 3)

    elif name == "FastAPI":
        _fill_poly(bitmap, [
            (58 * sx, 8 * sy), (28 * sx, 52 * sy), (48 * sx, 52 * sy),
            (42 * sx, 92 * sy), (72 * sx, 48 * sy), (52 * sx, 48 * sy),
        ])

    elif name == "LangChain":
        # Three overlapping rings (thick outlines)
        for cx in [22, 50, 78]:
            _fill_ellipse(bitmap, int(cx * sx), int(50 * sy), int(14 * sx), int(8 * sy))
            _fill_ellipse(bitmap, int(cx * sx), int(50 * sy), int(10 * sx), int(4 * sy), val=0)

    elif name == "Docker":
        # Containers
        for row_y in [28, 42]:
            for col_x in [22, 42, 62]:
                x0, y0 = int(col_x * sx), int(row_y * sy)
                x1, y1 = int((col_x + 16) * sx), int((row_y + 10) * sy)
                bitmap[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 1
        # Whale body ellipse
        _fill_ellipse(bitmap, int(50 * sx), int(65 * sy), int(30 * sx), int(12 * sy))


def _fill_poly(bitmap, pts, val=1):
    H, W = bitmap.shape
    if len(pts) < 3:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    y_min = max(0, int(min(ys)))
    y_max = min(H - 1, int(max(ys)))
    for y in range(y_min, y_max + 1):
        intersections = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                if y2 - y1 != 0:
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x)
        intersections.sort()
        for i in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(intersections[i]))
            x_end = min(W - 1, int(intersections[i + 1]))
            bitmap[y, x_start:x_end + 1] = val


def _fill_circle(bitmap, cx, cy, r, val=1):
    H, W = bitmap.shape
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                bitmap[y, x] = val


def _fill_ellipse(bitmap, cx, cy, rx, ry, val=1):
    H, W = bitmap.shape
    for y in range(max(0, cy - ry), min(H, cy + ry + 1)):
        for x in range(max(0, cx - rx), min(W, cx + rx + 1)):
            if (x - cx) ** 2 / max(1, rx ** 2) + (y - cy) ** 2 / max(1, ry ** 2) <= 1:
                bitmap[y, x] = val


def _draw_line_thick(bitmap, x0, y0, x1, y1, thick, val=1):
    H, W = bitmap.shape
    n_steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(n_steps + 1):
        t = i / n_steps
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        half = thick // 2
        bitmap[max(0, y - half):min(H, y + half + 1),
               max(0, x - half):min(W, x + half + 1)] = val


def optimal_transport_match(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Greedy approximation of optimal transport matching.
    Returns permutation index array: dst[perm[i]] is the target for src[i].
    Uses a random-project sort (fast, O(n log n), decent quality).
    """
    assert len(src) == len(dst)
    n = len(src)
    # Random projection direction
    theta = np.random.uniform(0, 2 * math.pi)
    proj_dir = np.array([math.cos(theta), math.sin(theta)])
    src_proj = src @ proj_dir
    dst_proj = dst @ proj_dir
    src_order = np.argsort(src_proj)
    dst_order = np.argsort(dst_proj)
    perm = np.empty(n, dtype=int)
    perm[src_order] = dst_order
    return perm


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — SVG ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def build_intro_animation(
    positions: list,
    intro_groups: list[list[int]],
    total_intro: float,
    fade_start: float,
    fade_end: float,
) -> dict[int, tuple]:
    """
    Returns {dot_idx: (begin_s, dur_s)} for each dot's intro fade-in.
    Groups are staggered over [fade_start, fade_end].
    """
    n_groups = len(intro_groups)
    timing = {}
    for g_idx, group in enumerate(intro_groups):
        t_begin = fade_start + g_idx * (fade_end - fade_start) / n_groups
        t_dur = 0.15 + random.uniform(-0.03, 0.03)  # slight jitter
        for dot_idx in group:
            timing[dot_idx] = (t_begin, t_dur)
    return timing


def _fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def build_portrait_group_svg(
    positions: list,
    dot_r: float,
    intro_timing: dict,
    drift_bands: list[list[int]],
    logo_centroid: tuple,
    drift_fraction: float,
    color: str,
    dark_mode: bool,
    loop_dur: float,
    portrait_hold: float,
) -> str:
    """
    Build the SVG <g> element containing all portrait dots with
    intro fade-in (per intro-group) and loop drift (per drift-band).

    Key optimisation: animate the GROUP, not each dot.
    - One <g> per intro-group -> one <animate opacity> per group (~60 total)
    - One <g> per drift-band  -> one <animateTransform> per band (~94 total)
    This cuts file size from ~17MB to ~400KB vs. per-dot SMIL.
    """
    r = dot_r
    lx, ly = logo_centroid

    # ── pre-compute drift offsets per band ───────────────────────────────────
    band_offsets = []
    for band in drift_bands:
        if not band:
            band_offsets.append((0.0, 0.0))
            continue
        cx_b = float(np.mean([positions[i][0] for i in band]))
        cy_b = float(np.mean([positions[i][1] for i in band]))
        band_offsets.append(((lx - cx_b) * drift_fraction,
                              (ly - cy_b) * drift_fraction))

    dot_to_band: dict[int, int] = {}
    for b_idx, band in enumerate(drift_bands):
        for dot_idx in band:
            dot_to_band[dot_idx] = b_idx

    # ── group dots by intro-group (from intro_timing) ───────────────────────
    # intro_timing: {dot_idx: (begin_s, dur_s)}
    # We need reverse map: (begin_s, dur_s) -> [dot_idx]
    from collections import defaultdict
    intro_group_map: dict[tuple, list[int]] = defaultdict(list)
    for dot_idx in range(len(positions)):
        key = intro_timing.get(dot_idx, (0.0, 0.15))
        intro_group_map[key].append(dot_idx)

    # ── keyframe parameters (loop) ───────────────────────────────────────────
    t_hold     = portrait_hold / loop_dur
    t_drift_out = (portrait_hold + 0.5) / loop_dur
    t_drift_in  = (loop_dur - 0.5) / loop_dur
    kt_loop = f"0;{t_hold:.3f};{t_drift_out:.3f};{t_drift_in:.3f};1"

    # ── build a band -> [dot_idx] lookup for the inner loop ─────────────────
    band_dots: dict[int, list[int]] = defaultdict(list)
    for dot_idx in range(len(positions)):
        band_dots[dot_to_band.get(dot_idx, 0)].append(dot_idx)

    out = ['<g id="portrait-layer" shape-rendering="crispEdges">']

    # Emit one <g> per intro-group (handles fade-in opacity animation)
    for (begin_s, dur_s), group_dot_ids in intro_group_map.items():
        # Within this intro group, further split by drift band
        # so the drift transform can be applied at band level.
        # Structure: <g opacity animate> -> <g animateTransform> -> <rect>s

        # band -> dots that are in this intro group AND this band
        ig_band: dict[int, list[int]] = defaultdict(list)
        for dot_idx in group_dot_ids:
            ig_band[dot_to_band.get(dot_idx, 0)].append(dot_idx)

        # Outer group: intro fade-in
        out.append(
            f'<g opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1"'
            f' begin="{begin_s:.3f}s" dur="{dur_s:.3f}s" fill="freeze"/>'
        )

        loop_begin = begin_s + dur_s  # loop starts after intro completes

        for b_idx, dot_ids in ig_band.items():
            dox, doy = band_offsets[b_idx]
            tx_vals = f"0,0;0,0;{dox:.2f},{doy:.2f};{dox:.2f},{doy:.2f};0,0"

            # Inner group: drift animation
            out.append(
                f'<g>'
                f'<animateTransform attributeName="transform" type="translate"'
                f' values="{tx_vals}" keyTimes="{kt_loop}"'
                f' dur="{loop_dur:.1f}s" begin="{loop_begin:.3f}s"'
                f' repeatCount="indefinite" additive="sum"/>'
            )

            # Static rects — no per-dot animation
            for dot_idx in dot_ids:
                px, py = positions[dot_idx]
                out.append(
                    f'<rect x="{px - r:.2f}" y="{py - r:.2f}"'
                    f' width="{2*r:.2f}" height="{2*r:.2f}" fill="{color}"/>'
                )

            out.append('</g>')  # close inner drift group

        out.append('</g>')  # close outer intro group

    out.append('</g>')  # close portrait-layer
    return "\n".join(out)


def build_travellers_svg(
    logo_positions: dict,  # logo_name -> (n,2) array
    n_travellers: int,
    loop_dur: float,
    portrait_hold: float,
    logo_hold: float,
    logo_trans: float,
    logo_order: list,
    color: str,
    dot_r: float,
) -> str:
    """
    Build the traveller swarm SVG with optimal-transport morphing.
    """
    r = dot_r * 1.8  # travellers are slightly larger
    n = n_travellers
    lines = [f'<g id="traveller-layer" shape-rendering="crispEdges">']

    # Build per-logo point arrays
    arrays = [logo_positions[name] for name in logo_order]

    # Optimal-transport matching chain
    perms = []
    for i in range(len(arrays) - 1):
        perms.append(optimal_transport_match(arrays[i], arrays[i + 1]))
    # Close the loop
    perms.append(optimal_transport_match(arrays[-1], arrays[0]))

    # Build timeline
    # Logo 0 starts at portrait_hold, each logo holds logo_hold, transition logo_trans
    phase_times = []
    t = portrait_hold
    for i in range(len(logo_order)):
        phase_times.append(t)
        t += logo_hold + logo_trans
    # Normalise to [0,1] for keyTimes
    # Opacity: travellers are invisible during portrait phase, visible during logos

    n_phases = len(logo_order)
    # Build keyTimes and values for each dot
    for dot_i in range(n):
        # Build position keyframes
        # For each logo phase, the dot is at arrays[logo_idx][mapped_idx]
        # We need to trace each dot's position through OT matchings

        # Find position at each logo
        pos_at_logo = [None] * n_phases
        pos_at_logo[0] = arrays[0][dot_i % len(arrays[0])]
        cur = dot_i % len(arrays[0])
        for li in range(1, n_phases):
            nxt = perms[li - 1][cur % len(perms[li - 1])]
            pos_at_logo[li] = arrays[li][nxt % len(arrays[li])]
            cur = nxt

        # Build SVG animate element
        kts = []
        tx_vals = []
        ty_vals = []
        op_vals = []

        # t=0: hidden (portrait phase)
        kts.append("0")
        tx_vals.append(f"{pos_at_logo[0][0]:.1f}")
        ty_vals.append(f"{pos_at_logo[0][1]:.1f}")
        op_vals.append("0")

        # portrait hold end
        t_hold_n = portrait_hold / loop_dur
        kts.append(f"{t_hold_n:.3f}")
        tx_vals.append(f"{pos_at_logo[0][0]:.1f}")
        ty_vals.append(f"{pos_at_logo[0][1]:.1f}")
        op_vals.append("0")

        # Each logo phase
        for li, logo_name in enumerate(logo_order):
            t_start = phase_times[li] / loop_dur
            t_end = (phase_times[li] + logo_hold) / loop_dur
            t_trans = min(1.0, (phase_times[li] + logo_hold + logo_trans) / loop_dur)

            # Fade in
            kts.append(f"{min(t_start + 0.05, t_end):.3f}")
            tx_vals.append(f"{pos_at_logo[li][0]:.1f}")
            ty_vals.append(f"{pos_at_logo[li][1]:.1f}")
            op_vals.append("1")

            # Hold
            kts.append(f"{t_end:.3f}")
            tx_vals.append(f"{pos_at_logo[li][0]:.1f}")
            ty_vals.append(f"{pos_at_logo[li][1]:.1f}")
            op_vals.append("1")

            # Transition
            next_li = (li + 1) % n_phases
            kts.append(f"{t_trans:.3f}")
            tx_vals.append(f"{pos_at_logo[next_li][0]:.1f}")
            ty_vals.append(f"{pos_at_logo[next_li][1]:.1f}")
            op_vals.append("0" if t_trans >= 0.999 else "1")

        kts.append("1")
        tx_vals.append(f"{pos_at_logo[0][0]:.1f}")
        ty_vals.append(f"{pos_at_logo[0][1]:.1f}")
        op_vals.append("0")

        kt_str = ";".join(kts)
        tx_str = ";".join(tx_vals)
        ty_str = ";".join(ty_vals)
        op_str = ";".join(op_vals)
        cx = pos_at_logo[0][0]
        cy = pos_at_logo[0][1]

        elem = (
            f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" '
            f'width="{2*r:.1f}" height="{2*r:.1f}" '
            f'fill="{color}" opacity="0">'
            f'<animate attributeName="cx" values="{tx_str}" keyTimes="{kt_str}" '
            f'dur="{loop_dur:.1f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{ty_str}" keyTimes="{kt_str}" '
            f'dur="{loop_dur:.1f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{op_str}" keyTimes="{kt_str}" '
            f'dur="{loop_dur:.1f}s" repeatCount="indefinite"/>'
            f'</rect>'
        )
        lines.append(elem)

    lines.append('</g>')
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# INFO PANEL SVG
# ════════════════════════════════════════════════════════════════════════════

def build_info_panel_svg(dark: bool, ox: float, oy: float, w: float, h: float) -> str:
    """Build the SYSTEM.INFO terminal panel SVG."""
    bg = CFG['PANEL_DARK'] if dark else CFG['PANEL_LIGHT']
    chrome = CFG['CHROME_DARK'] if dark else CFG['CHROME_LIGHT']
    text_col = CFG['TEXT_DARK'] if dark else CFG['TEXT_LIGHT']
    muted = CFG['MUTED_DARK'] if dark else CFG['MUTED_LIGHT']
    accent = CFG['ACCENT']
    danger = CFG['DANGER']
    purple = '#7C3AED'
    green = CFG['ACCENT']

    lines = []
    # Panel background
    lines.append(
        f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="6" '
        f'fill="{bg}" stroke="{chrome}" stroke-width="1"/>'
    )
    # Title bar
    lines.append(
        f'<rect x="{ox}" y="{oy}" width="{w}" height="32" rx="6" fill="{"#060C17" if dark else "#F1F5F9"}"/>'
    )
    lines.append(
        f'<rect x="{ox}" y="{oy+20}" width="{w}" height="12" fill="{"#060C17" if dark else "#F1F5F9"}"/>'
    )
    # Traffic lights
    for i, c in enumerate(['#EF4444', '#F59E0B', '#10B981']):
        lines.append(f'<circle cx="{ox+16+i*16}" cy="{oy+16}" r="5" fill="{c}"/>')
    # Title text
    lines.append(
        f'<text x="{ox+w/2}" y="{oy+20}" text-anchor="middle" '
        f'fill="{muted}" font-size="11" font-family="Share Tech Mono, monospace">'
        f'profile.sh --live</text>'
    )
    # LIVE badge
    bx, by = ox + w - 55, oy + 8
    lines.append(f'<rect x="{bx}" y="{by}" width="46" height="18" rx="9" fill="{danger}" opacity="0.15"/>')
    lines.append(f'<rect x="{bx}" y="{by}" width="46" height="18" rx="9" fill="none" stroke="{danger}" stroke-width="0.8"/>')
    lines.append(
        f'<circle cx="{bx+11}" cy="{by+9}" r="4" fill="{danger}">'
        f'<animate attributeName="opacity" values="1;0.15;1" dur="1.2s" repeatCount="indefinite"/>'
        f'</circle>'
    )
    lines.append(
        f'<text x="{bx+25}" y="{by+13}" text-anchor="middle" '
        f'fill="{danger}" font-size="9" font-weight="bold" '
        f'font-family="Share Tech Mono, monospace">LIVE</text>'
    )
    # Divider
    lines.append(f'<line x1="{ox}" y1="{oy+32}" x2="{ox+w}" y2="{oy+32}" stroke="{chrome}" stroke-width="0.5" opacity="0.4"/>')

    # Corner brackets
    bw = 16
    for bx_c, by_c, dx, dy in [
        (ox+2, oy+2, 1, 1), (ox+w-2, oy+2, -1, 1),
        (ox+2, oy+h-2, 1, -1), (ox+w-2, oy+h-2, -1, -1)
    ]:
        lines.append(
            f'<path d="M {bx_c},{by_c+dy*bw} L {bx_c},{by_c} L {bx_c+dx*bw},{by_c}" '
            f'stroke="{chrome}" stroke-width="1.5" fill="none" opacity="0.9"/>'
        )

    # Prompt line
    py_cur = oy + 50
    row_h = 22
    fm = dict(family="Share Tech Mono, monospace", size=12)

    def text(x, y, content, color, size=12, anchor="start", bold=False):
        fw = "bold" if bold else "normal"
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'fill="{color}" font-size="{size}" font-weight="{fw}" '
            f'font-family="Share Tech Mono, monospace">{content}</text>'
        )

    lines.append(text(ox+12, py_cur, "root@nexus", green, 12))
    lines.append(text(ox+88, py_cur, ":", muted, 12))
    lines.append(text(ox+96, py_cur, "~/profile", chrome, 12))
    lines.append(text(ox+172, py_cur, "$ cat system.info", text_col, 12))
    py_cur += row_h

    # Divider row
    lines.append(text(ox+12, py_cur, "-" * 44, muted, 11))
    py_cur += row_h

    for label, value in INFO_ROWS:
        if label is None:
            lines.append(text(ox+12, py_cur, "-" * 44, muted, 11))
            py_cur += row_h
            continue
        # Leader dots
        max_label = 13
        dots = "." * (max_label - len(label) + 4)
        lines.append(text(ox+12, py_cur, label, purple, 12))
        lines.append(text(ox+12+len(label)*7.5, py_cur, dots, muted, 12))
        val_x = ox + 12 + (max_label + 6) * 7.5
        val_color = chrome if label in ("Role", "Grid.Port", "Grid.LinkedIn", "Grid.GitHub") else text_col
        if label == "Status":
            val_color = accent
        lines.append(text(val_x, py_cur, value, val_color, 12))
        py_cur += row_h

    # Blinking cursor
    lines.append(text(ox+12, py_cur, "root@nexus", green, 12))
    lines.append(text(ox+88, py_cur, ":", muted, 12))
    lines.append(text(ox+96, py_cur, "~/profile", chrome, 12))
    lines.append(text(ox+172, py_cur, "$ ", text_col, 12))
    lines.append(
        f'<rect x="{ox+188}" y="{py_cur-12}" width="8" height="13" fill="{chrome}">'
        f'<animate attributeName="opacity" values="1;0;1" dur="1s" calcMode="discrete" repeatCount="indefinite"/>'
        f'</rect>'
    )

    # Handle pill
    py_cur += row_h + 10
    pill_w, pill_h = 160, 26
    pill_x = ox + (w - pill_w) / 2
    lines.append(f'<rect x="{pill_x}" y="{py_cur}" width="{pill_w}" height="{pill_h}" rx="13" fill="{chrome}" opacity="0.15"/>')
    lines.append(f'<rect x="{pill_x}" y="{py_cur}" width="{pill_w}" height="{pill_h}" rx="13" fill="none" stroke="{chrome}" stroke-width="0.8"/>')
    lines.append(text(ox+w/2, py_cur+17, "@theshauryas1", chrome, 13, anchor="middle", bold=True))

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# FULL SVG BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_full_svg(
    portrait_svg_group: str,
    travellers_svg_group: str,
    info_panel_svg: str,
    dark: bool,
    logo_svg_overlays: str,
) -> str:
    bg = CFG['BG_DARK'] if dark else CFG['BG_LIGHT']
    chrome = CFG['CHROME_DARK'] if dark else CFG['CHROME_LIGHT']
    panel = CFG['PANEL_DARK'] if dark else CFG['PANEL_LIGHT']
    W = CFG['SVG_W']
    H = CFG['SVG_H']

    portrait_panel_x = 28
    portrait_panel_y = 22
    portrait_panel_w = 320
    portrait_panel_h = 566

    info_panel_x = portrait_panel_x + portrait_panel_w + 24
    info_panel_y = 22
    info_panel_w = W - info_panel_x - 28
    info_panel_h = 566

    grid_color = chrome

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Share Tech Mono', 'Courier New', monospace; }}
  </style>
  <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%"  stop-color="{bg}"/>
    <stop offset="100%" stop-color="{"#060C17" if dark else "#E2EEF8"}"/>
  </linearGradient>
  <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%"   stop-color="{chrome}"/>
    <stop offset="100%" stop-color="{"#06B6D4" if dark else "#0E7490"}"/>
  </linearGradient>
  <pattern id="gridPat" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="{chrome}" stroke-width="0.3" opacity="0.08"/>
  </pattern>
</defs>

<!-- Background -->
<rect width="{W}" height="{H}" fill="url(#bgGrad)"/>
<rect width="{W}" height="{H}" fill="url(#gridPat)"/>
<rect x="0" y="0" width="{W}" height="2" fill="url(#cyanGrad)" opacity="0.9"/>
<rect x="0" y="{H-2}" width="{W}" height="2" fill="url(#cyanGrad)" opacity="0.9"/>

<!-- Portrait Panel -->
<rect x="{portrait_panel_x}" y="{portrait_panel_y}"
      width="{portrait_panel_w}" height="{portrait_panel_h}"
      rx="6" fill="{panel}" stroke="{chrome}" stroke-width="1" opacity="0.95"/>

<!-- Portrait dots (both layers) -->
{portrait_svg_group}
{travellers_svg_group}

<!-- Logo label overlays -->
{logo_svg_overlays}

<!-- Info Panel -->
{info_panel_svg}

<!-- Bottom status bar -->
<text x="30" y="{H-8}" fill="{chrome if dark else "#0891B2"}" font-size="10" font-family="Share Tech Mono, monospace">
  [ AI/ML ENGINEER ] ● OPEN TO OPPORTUNITIES ● theshauryas1 ● India ● NEXUSSWARM // INFRAMIND AI
</text>
</svg>
'''
    return svg


# ════════════════════════════════════════════════════════════════════════════
# LOGO LABEL OVERLAYS (shown during traveller phase)
# ════════════════════════════════════════════════════════════════════════════

def build_logo_overlays(
    logo_order: list,
    portrait_hold: float,
    logo_hold: float,
    logo_trans: float,
    loop_dur: float,
    cx: float, cy: float,
    dark: bool,
) -> str:
    """Build logo name + subtitle labels that appear during each logo phase."""
    lines = []
    t = portrait_hold
    n = len(logo_order)
    for i, name in enumerate(logo_order):
        logo = LOGOS[name]
        t_begin = t
        t_end = t + logo_hold
        t_out = t + logo_hold + logo_trans
        t_begin_n = t_begin / loop_dur
        t_end_n = t_end / loop_dur
        t_out_n = min(1.0, t_out / loop_dur)

        opacity_vals = f"0;0;1;1;0;0"
        kt = f"0;{t_begin_n:.3f};{min(t_begin_n+0.05,t_end_n):.3f};{t_end_n:.3f};{t_out_n:.3f};1"

        lines.append(
            f'<g opacity="0">'
            f'<animate attributeName="opacity" values="{opacity_vals}" keyTimes="{kt}" '
            f'dur="{loop_dur:.1f}s" repeatCount="indefinite"/>'
            f'<text x="{cx}" y="{cy+60}" text-anchor="middle" '
            f'fill="{logo["color"]}" font-size="18" font-weight="bold" '
            f'font-family="Orbitron, Share Tech Mono, monospace">{logo["label"]}</text>'
            f'<text x="{cx}" y="{cy+80}" text-anchor="middle" '
            f'fill="{"#94A3B8" if dark else "#64748B"}" font-size="11" '
            f'font-family="Share Tech Mono, monospace">{logo["sub"]}</text>'
            f'</g>'
        )
        t += logo_hold + logo_trans

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def generate(photo_path: Path, out_dir: Path):
    random.seed(42)
    np.random.seed(42)

    print(f"[1/8] Loading photo: {photo_path}")
    img = Image.open(photo_path).convert("RGB")
    img = preprocess(img)
    print(f"      Preprocessed -> {img.size}")

    print("[2/8] Background segmentation ...")
    fg_mask = get_fg_mask(img)
    coverage = fg_mask.sum() / fg_mask.size * 100
    print(f"      Foreground coverage: {coverage:.1f}%")

    print("[3/8] Floyd-Steinberg dither ...")
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    dither_full = floyd_steinberg(gray)
    # Dark mode: only subject dots
    dither_dark = dither_full * fg_mask
    # Light mode: all dots
    dither_light = dither_full
    ink_dark = dither_dark.sum()
    ink_light = dither_light.sum()
    print(f"      Dark ink: {ink_dark:.0f} dots | Light ink: {ink_light:.0f} dots")

    # Portrait panel geometry
    px_offset = 36.0   # x offset within SVG where portrait dots start
    py_offset = 30.0   # y offset
    panel_cx = px_offset + CFG['GRID_W'] / 2
    panel_cy = py_offset + CFG['GRID_H'] / 2

    print("[4/8] Collecting dot positions ...")
    positions_dark = collect_dot_positions(dither_dark, px_offset, py_offset)
    positions_light = collect_dot_positions(dither_light, px_offset, py_offset)
    print(f"      Dark: {len(positions_dark)} | Light: {len(positions_light)}")

    print("[5/8] Building intro animation groups ...")
    intro_groups_dark = make_intro_groups(positions_dark, 60)
    intro_groups_light = make_intro_groups(positions_light, 60)
    ev_dark = check_intro_evenness(positions_dark, intro_groups_dark)
    ev_light = check_intro_evenness(positions_light, intro_groups_light)
    print(f"      Evenness sigma dark: {ev_dark:.4f} (target <0.05) | light: {ev_light:.4f}")

    intro_timing_dark = build_intro_animation(
        positions_dark, intro_groups_dark,
        CFG['INTRO_DUR'], CFG['INTRO_FADE_START'], CFG['INTRO_FADE_END']
    )
    intro_timing_light = build_intro_animation(
        positions_light, intro_groups_light,
        CFG['INTRO_DUR'], CFG['INTRO_FADE_START'], CFG['INTRO_FADE_END']
    )

    print("[6/8] Building drift bands ...")
    logo_centroid = (panel_cx, panel_cy)
    drift_dark = make_drift_bands(
        positions_dark, CFG['N_DRIFT_BANDS'], logo_centroid,
        CFG['DRIFT_FRACTION'], CFG['DRIFT_NOISE_SIGMA']
    )
    drift_light = make_drift_bands(
        positions_light, CFG['N_DRIFT_BANDS'], logo_centroid,
        CFG['DRIFT_FRACTION'], CFG['DRIFT_NOISE_SIGMA']
    )
    sb_dark = check_straight_boundary(positions_dark, drift_dark)
    print(f"      Straight-boundary metric: {sb_dark:.2f} (std of band labels; organic if >5 with noise)")

    print("[7/8] Sampling logo traveller positions ...")
    logo_positions = {}
    n_trav = CFG['N_TRAVELLERS']
    # Use the portrait panel area for logo rendering
    for name in LOGO_ORDER:
        pts = sample_logo_dots(name, n_trav, px_offset, py_offset,
                               float(CFG['GRID_W']), float(CFG['GRID_H']))
        logo_positions[name] = pts
        print(f"      {name}: {len(pts)} travellers")

    print("[8/8] Assembling SVGs ...")
    for dark in [True, False]:
        mode = "dark" if dark else "light"
        positions = positions_dark if dark else positions_light
        intro_timing = intro_timing_dark if dark else intro_timing_light
        drift_bands = drift_dark if dark else drift_light
        portrait_color = CFG['PORTRAIT_HUE'] if dark else CFG['PORTRAIT_HUE_LIGHT']
        chrome = CFG['CHROME_DARK'] if dark else CFG['CHROME_LIGHT']

        portrait_group = build_portrait_group_svg(
            positions, CFG['DOT_R'], intro_timing, drift_bands,
            logo_centroid, CFG['DRIFT_FRACTION'],
            portrait_color, dark,
            CFG['LOOP_DUR'], CFG['PORTRAIT_HOLD'],
        )

        travellers_group = build_travellers_svg(
            logo_positions, n_trav,
            CFG['LOOP_DUR'], CFG['PORTRAIT_HOLD'],
            CFG['LOGO_HOLD'], CFG['LOGO_TRANS'],
            LOGO_ORDER, portrait_color, CFG['DOT_R'],
        )

        logo_overlays = build_logo_overlays(
            LOGO_ORDER, CFG['PORTRAIT_HOLD'], CFG['LOGO_HOLD'],
            CFG['LOGO_TRANS'], CFG['LOOP_DUR'],
            panel_cx, panel_cy + 150, dark,
        )

        info_panel = build_info_panel_svg(
            dark,
            ox=372.0, oy=22.0,
            w=CFG['SVG_W'] - 372.0 - 28.0,
            h=566.0,
        )

        full_svg = build_full_svg(
            portrait_group, travellers_group, info_panel, dark, logo_overlays
        )

        out_path = out_dir / f"{mode}.svg"
        out_path.write_text(full_svg, encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"      Written: {out_path}  ({size_kb:.0f} KB)")

    print("\n? Done! Open dark.svg / light.svg in a browser to verify.")
    print("   Tip: Use ?v=999 suffix on raw.githubusercontent.com to bust CDN cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animated GitHub profile banner")
    parser.add_argument("--photo", type=Path, required=True, help="Path to portrait photo")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    args = parser.parse_args()

    if not args.photo.exists():
        print(f"ERROR: Photo not found: {args.photo}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate(args.photo, args.out_dir)
