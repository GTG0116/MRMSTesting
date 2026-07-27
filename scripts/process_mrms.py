import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gzip
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from contourpy import contour_generator, FillType
from scipy.ndimage import (gaussian_filter, maximum_filter, grey_closing,
                           binary_fill_holes, binary_opening)
from datetime import datetime, timezone, timedelta
import pytz
import gc

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 55.0, 20.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 15
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# Physically-based rate cap for wintry precipitation (mm/hr liquid equivalent).
# Real snowfall tops out ~5 mm/hr liquid; real ice pellets ~8 mm/hr.
# Anything above this threshold is almost certainly misclassified hail or
# heavy convective rain — force those pixels to the rain layer.
WINTRY_RATE_MAX = 15.0  # mm/hr (internal threshold, raw data units)

# Unit conversion: all display values are in inches
MM_TO_IN = 1.0 / 25.4

# --- CONTOUR SMOOTHING ---
# Every frame is published as smoothed filled contours instead of a pixel
# raster: the field is coarsened, Gaussian-smoothed, and traced into filled
# contour bands. The bands are written as GeoJSON polygons, so the browser
# draws real vector shapes — they stay smooth at any zoom instead of breaking
# up into image pixels, and only the fill is drawn (no contour outlines).
#
# Coarsening takes the block MAXIMUM rather than the mean: averaging a small,
# intense echo against the empty background drags its peak below the lowest
# colour bound and the feature disappears entirely.
#
# Two things keep the bands looking like weather rather than like a mesh of
# polygons: narrow blank pockets inside an echo are closed before the blur, so
# a storm doesn't come out riddled with holes next to its core, and every ring
# is put in a canonical order before it is simplified, so the boundary two
# neighbouring bands share simplifies to exactly the same points and no gap
# opens up between them.
SMOOTH_SIGMA      = 1.5  # Gaussian sigma, in coarse-grid cells
CONTOUR_DOWNSCALE = 2    # block-max factor applied before contouring
SMOOTH_DILATE     = 1    # max-filter width (coarse cells) applied before blurring;
                         # >1 keeps speckly fields from being blurred away
SMOOTH_GAP_FILL   = 4    # close blank pockets up to this many coarse cells wide
                         # before blurring, so storms don't come out with holes

# GeoJSON output tuning. Simplify tolerance and minimum polygon area are in
# coarse cells (one coarse cell ≈ 2 km); the contours are already smoothed at
# a larger scale than that, so simplification costs no visible detail while
# cutting the payload by ~3x.
GEOJSON_SIMPLIFY  = 0.75  # Douglas-Peucker tolerance, in coarse cells
GEOJSON_MIN_AREA  = 1.0   # drop polygons smaller than this many coarse cells²
GEOJSON_DECIMALS  = 4     # coordinate precision (~11 m)
GEOJSON_ROUND     = 1     # Chaikin corner-cutting passes over each ring; rounds
                          # off the facets simplification leaves behind

# --- SESSION SETUP ---
session = requests.Session()
retry = Retry(connect=3, read=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# --- MERCATOR MATH ---
def lat_to_merc(lat):
    """Converts latitude to normalised Mercator Y."""
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

def merc_to_lat(y):
    """Converts normalised Mercator Y back to latitude."""
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

# --- COLOR TABLES: PRECIPITATION RATE (all bounds in in/hr) ---
RAIN_BOUNDS = [0.01, 0.02, 0.06, 0.12, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00]  # in/hr
SNOW_BOUNDS = [0.004, 0.01, 0.02, 0.04, 0.10, 0.24]                                 # in/hr
ICE_BOUNDS  = [0.004, 0.01, 0.02, 0.04, 0.10, 0.24]                                 # in/hr

RAIN_COLORS = [
    '#00fb90',  # 0.01 – 0.02  in/hr : mint / light green
    '#00cc00',  # 0.02 – 0.06  in/hr : medium-light green
    '#009900',  # 0.06 – 0.12  in/hr : medium green
    '#006600',  # 0.12 – 0.20  in/hr : dark green
    '#ffff00',  # 0.20 – 0.50  in/hr : bright yellow
    '#ffcc00',  # 0.50 – 1.00  in/hr : amber
    '#ff9100',  # 1.00 – 1.50  in/hr : orange
    '#ff5500',  # 1.50 – 2.00  in/hr : red-orange
    '#ff0000',  # 2.00 – 3.00  in/hr : red
    '#cc0000',  # 3.00 – 5.00  in/hr : dark red
]
SNOW_COLORS = ['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff']
ICE_COLORS  = ['#ff00ff', '#d100d1', '#910091', '#4b0082', '#2d004b']

def _disc(width):
    """Boolean disc footprint `width` cells across (round, not square)."""
    r = (width - 1) / 2.0
    o = np.arange(width) - r
    return np.hypot(o[:, None], o[None, :]) <= r + 0.25


def _fill_gaps(grid, threshold, width):
    """Close narrow blank pockets sitting inside an echo.

    MRMS drops out in patches inside storms — beam blockage, the cone of
    silence right over a radar, cells the rate algorithm zeroes out, pixels
    this layer's type mask handed to one of the other two layers. Those
    patches sit below the lowest colour bound, survive the blur, and the
    contours punch a blank hole through the storm — often right beside its
    most intense part, where the eye is least willing to accept it.

    Only cells that are blank (below `threshold`), enclosed by the echo, and
    part of a pocket narrower than `width` cells are touched; they take the
    value of a grey-scale closing, i.e. of the data that surrounds them.
    Everything else — the intensity structure of the storm, the outer edge of
    the echo, and wide gaps that are genuinely precipitation-free — is left
    exactly as it was.
    """
    if width < 2:
        return grid

    blank = grid < threshold
    if not blank.any():
        return grid

    disc     = _disc(int(width))
    enclosed = binary_fill_holes(~blank)          # the echo and what it surrounds
    narrow   = blank & ~binary_opening(blank, structure=disc)
    pockets  = enclosed & narrow
    if not pockets.any():
        return grid

    return np.where(pockets, grey_closing(grid, footprint=disc, mode='nearest'),
                    grid)


def _smooth_field(vals, floor, gap_below=None, sigma=SMOOTH_SIGMA,
                  downscale=CONTOUR_DOWNSCALE, dilate=SMOOTH_DILATE,
                  gap_fill=SMOOTH_GAP_FILL):
    """Coarsen and Gaussian-smooth a field so it can be contoured.

    No-data cells (NaN) are replaced with `floor` — a value below the lowest
    colour bound — so the smoothed field tapers away to nothing at the edge of
    the echo rather than ending on a hard, blocky boundary. No-data pockets
    *inside* an echo are then closed up (see `_fill_gaps`) so they don't turn
    into holes in the middle of a storm.

    Returns (grid, x, y) where x/y are the pixel coordinates of the coarse
    cell centres on the original full-resolution grid.
    """
    grid = np.where(np.isfinite(vals), vals, floor).astype(np.float32)

    d = max(1, int(downscale))
    if d > 1:
        h, w   = grid.shape
        hc, wc = (h // d) * d, (w // d) * d
        grid   = grid[:hc, :wc].reshape(hc // d, d, wc // d, d).max(axis=(1, 3))

    if dilate > 1:
        grid = maximum_filter(grid, size=int(dilate))

    if gap_below is not None:
        grid = _fill_gaps(grid, gap_below, gap_fill)

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma, mode='nearest')

    x = (np.arange(grid.shape[1]) + 0.5) * d
    y = (np.arange(grid.shape[0]) + 0.5) * d
    return grid, x, y


def _contour_levels(bounds, colors, min_val=None):
    """Pick the contour levels and their fill colours.

    `min_val` is the threshold the data was masked at. Below the first bound
    it adds a sub-threshold sliver in the first colour, so near-threshold
    pixels still show. Above it, the bands the data can never reach are
    dropped so the smoothed halo cannot spill into them.

    The returned colour list has one extra entry for the open-ended top band,
    which reuses the top colour (what BoundaryNorm's over-colour did).
    """
    levels      = list(bounds)
    fill_colors = list(colors)
    if min_val is not None:
        i = int(np.searchsorted(levels, min_val, side='right')) - 1
        if i < 0:
            levels      = [min_val] + levels
            fill_colors = [fill_colors[0]] + fill_colors
        elif 0 < i < len(fill_colors):
            levels      = [max(min_val, levels[i])] + levels[i + 1:]
            fill_colors = fill_colors[i:]
    return levels, fill_colors + [fill_colors[-1]]


def _ring_area(ring):
    """Absolute area of a closed ring, in the ring's own units."""
    rx, ry = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(rx, np.roll(ry, 1)) - np.dot(ry, np.roll(rx, 1)))


def _canonical_ring(ring):
    """Rewrite a closed ring in a canonical form: no repeated end point, wound
    counter-clockwise, starting at its lowest-leftmost vertex.

    Neighbouring bands share their boundaries — the curve that closes one band
    is the very same curve that opens the next — but contourpy hands the two
    copies back with different start vertices and opposite windings.
    Simplifying them as given picks different vertices to keep, and the sliver
    between the two versions shows through as a blank hairline, most visibly
    where the bands crowd together around an intense core. Canonicalising
    first makes both copies simplify to identical points, so the bands tile
    exactly.
    """
    pts = ring[:-1] if np.array_equal(ring[0], ring[-1]) else ring
    if len(pts) < 3:
        return pts
    signed = (np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) -
              np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
    if signed < 0:
        pts = pts[::-1]
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    return np.roll(pts, -start, axis=0)


def _round_ring(ring, passes=GEOJSON_ROUND):
    """Chaikin corner cutting on a closed ring.

    Simplification leaves long straight facets that read as a polygon rather
    than a weather feature; each pass replaces every corner with two points a
    quarter of the way along its edges, which rounds the outline off without
    moving it more than a fraction of a cell. The cut is symmetric, so a ring
    and its reverse round identically and shared band boundaries stay matched.
    """
    pts = ring[:-1]
    for _ in range(int(passes)):
        if len(pts) < 3:
            break
        nxt = np.roll(pts, -1, axis=0)
        cut = np.empty((len(pts) * 2, 2), dtype=pts.dtype)
        cut[0::2] = 0.75 * pts + 0.25 * nxt
        cut[1::2] = 0.25 * pts + 0.75 * nxt
        pts = cut
    return np.vstack([pts, pts[:1]])


def _simplify_ring(points, tol):
    """Douglas-Peucker simplification of an (N, 2) ring, iteratively."""
    n = len(points)
    if n < 3 or tol <= 0:
        return points
    keep       = np.zeros(n, dtype=bool)
    keep[0]    = keep[-1] = True
    stack      = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg    = points[a:b + 1]
        p0, p1 = points[a], points[b]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = np.hypot(dx, dy)
        if length == 0:
            dist = np.hypot(seg[:, 0] - p0[0], seg[:, 1] - p0[1])
        else:
            dist = np.abs(dx * (p0[1] - seg[:, 1]) -
                          (p0[0] - seg[:, 0]) * dy) / length
        i = int(np.argmax(dist))
        if dist[i] > tol:
            keep[a + i] = True
            stack.append((a, a + i))
            stack.append((a + i, b))
    return points[keep]


def _clean_ring(ring, tol, min_area):
    """Canonicalise, simplify and round one contour ring.

    Returns the finished closed ring, or None if it is too small to keep. The
    decision and the result depend only on the ring's geometry, never on how
    contourpy happened to hand it over, so the two bands that share a boundary
    always make the same call about it.
    """
    if len(ring) < 4 or _ring_area(ring) < min_area:
        return None
    pts = _canonical_ring(ring)
    if len(pts) < 3:
        return None
    closed = np.vstack([pts, pts[:1]])
    closed = _simplify_ring(closed, tol)
    if len(closed) < 4:
        return None
    closed[-1] = closed[0]  # simplification must not break closure
    return _round_ring(closed)


def contour_features(vals, bounds, colors, width_px, height_px,
                     sigma=SMOOTH_SIGMA, dilate=SMOOTH_DILATE, min_val=None):
    """Trace `vals` into smoothed filled contour bands as GeoJSON features.

    Each feature is one band polygon (with holes) carrying its fill colour in
    the 'c' property and the band's value range in 'v0'/'v1'.
    """
    levels, fill_colors = _contour_levels(bounds, colors, min_val)

    # Background floor: one full colour band below the lowest level, so the
    # smoothed halo around an echo drops out quickly instead of spreading the
    # fill outward. Keyed off the first real band (not the optional min_val
    # sliver, which can be far too narrow to separate data from background).
    floor      = levels[0] - (bounds[1] - bounds[0])
    grid, x, y = _smooth_field(vals, floor, gap_below=levels[0],
                               sigma=sigma, dilate=dilate)

    # Contour coordinates come back in full-resolution pixel space; convert to
    # lon/lat. Rows are evenly spaced in Mercator Y, so the inverse Mercator
    # gives back the true latitude of each row.
    merc_top, merc_bot = lat_to_merc(LAT_TOP), lat_to_merc(LAT_BOT)

    d        = max(1, int(CONTOUR_DOWNSCALE))
    tol      = GEOJSON_SIMPLIFY * d      # tolerance in full-res pixels
    min_area = GEOJSON_MIN_AREA * d * d

    cg       = contour_generator(x=x, y=y, z=grid,
                                 fill_type=FillType.ChunkCombinedOffsetOffset)
    features = []
    tops     = list(levels[1:]) + [np.inf]

    for k, (lower, upper) in enumerate(zip(levels, tops)):
        points_list, offsets_list, outer_list = cg.filled(lower, upper)[:3]
        for pts, offsets, outer in zip(points_list, offsets_list, outer_list):
            if pts is None:
                continue
            # `outer` groups rings into polygons: the first ring of each group
            # is the exterior, the rest are its holes — the order GeoJSON wants.
            for p in range(len(outer) - 1):
                rings = []
                for r in range(outer[p], outer[p + 1]):
                    ring = _clean_ring(pts[offsets[r]:offsets[r + 1]],
                                       tol, min_area)
                    if ring is None:
                        # A dropped exterior takes its holes with it: keeping
                        # them would promote a hole to the polygon's outline.
                        if r == outer[p]:
                            rings = []
                            break
                        continue
                    lon = LON_LEFT + ring[:, 0] / width_px * (LON_RIGHT - LON_LEFT)
                    lat = merc_to_lat(
                        merc_top - ring[:, 1] / height_px * (merc_top - merc_bot))
                    rings.append(np.stack([lon.round(GEOJSON_DECIMALS),
                                           lat.round(GEOJSON_DECIMALS)], 1).tolist())
                if rings:
                    features.append({
                        'type': 'Feature',
                        'properties': {
                            'c':  fill_colors[k],
                            'v0': round(float(lower), 6),
                            'v1': None if np.isinf(upper) else round(float(upper), 6),
                        },
                        'geometry': {'type': 'Polygon', 'coordinates': rings},
                    })
    return features


def save_geojson(features, path):
    """Write a FeatureCollection compactly (no spaces — these files are large)."""
    with open(path, 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f,
                  separators=(',', ':'))
    return os.path.getsize(path)


# --- COLOR TABLES: SINGLE-FIELD PRODUCTS ---
# Each entry: prefix, bounds (N+1 values for N color intervals), colors (N values),
# min_val (display threshold), max_val (fill-value cap).
# Optional 'conversion': multiply raw GRIB2 value (mm) by this factor before
# applying bounds/min_val/max_val. Use MM_TO_IN (1/25.4) for inch-display products.
SINGLE_PRODUCTS = {
    'mesh': {
        'prefix': 'CONUS/MESH_00.50',
        'bounds': [0.25, 0.50, 0.75, 1.00, 1.50, 1.75, 2.00, 2.50, 3.00],  # inches hail diameter
        'colors': [
            '#c8f500',  # 0.25–0.50 in (~pea)     : lime-yellow
            '#ffff00',  # 0.50–0.75 in (~marble)   : bright yellow
            '#ffd700',  # 0.75–1.00 in             : gold
            '#ff8c00',  # 1.00–1.50 in (~quarter)  : dark orange
            '#ff4500',  # 1.50–1.75 in             : red-orange
            '#ff0000',  # 1.75–2.00 in (~golf ball): red
            '#cc0000',  # 2.00–2.50 in             : dark red
            '#7f0000',  # 2.50–3.00 in (~baseball) : deep red
        ],
        'min_val': 0.20,   # inches (~5 mm)
        'max_val': 8.0,    # inches (~200 mm fill-value cap)
        'conversion': MM_TO_IN,
        # Hail swaths are narrow — smooth them lightly so they keep their shape.
        'smooth_sigma': 1.0,
        'label': 'Max Estimated Hail Size (in)',
    },
    'qpe6h': {
        'prefix': 'CONUS/RadarOnly_QPE_06H_00.00',
        'bounds': [0.05, 0.25, 0.50, 1.00, 2.00, 3.00, 4.00, 6.00, 8.00],  # inches accumulated
        'colors': [
            '#00fb90',  # 0.05–0.25 in : mint
            '#00cc00',  # 0.25–0.50 in : green
            '#009900',  # 0.50–1.00 in : medium green
            '#006600',  # 1.00–2.00 in : dark green
            '#ffff00',  # 2.00–3.00 in : yellow
            '#ffcc00',  # 3.00–4.00 in : amber
            '#ff9100',  # 4.00–6.00 in : orange
            '#ff0000',  # 6.00–8.00 in : red
        ],
        'min_val': 0.04,   # inches (~1 mm)
        'max_val': 40.0,   # inches (~1000 mm fill-value cap)
        'conversion': MM_TO_IN,
        'label': '6-Hour QPE (in)',
    },
    'qpe24h': {
        'prefix': 'CONUS/RadarOnly_QPE_24H_00.00',
        'bounds': [0.25, 1.00, 2.00, 3.00, 4.00, 6.00, 8.00, 12.00, 16.00],  # inches accumulated
        'colors': [
            '#00fb90',  # 0.25–1.00 in  : mint
            '#00cc00',  # 1.00–2.00 in  : green
            '#009900',  # 2.00–3.00 in  : medium green
            '#006600',  # 3.00–4.00 in  : dark green
            '#ffff00',  # 4.00–6.00 in  : yellow
            '#ffcc00',  # 6.00–8.00 in  : amber
            '#ff9100',  # 8.00–12.00 in : orange
            '#ff0000',  # 12.0–16.00 in : red
        ],
        'min_val': 0.20,   # inches (~5 mm)
        'max_val': 80.0,   # inches (~2000 mm fill-value cap)
        'conversion': MM_TO_IN,
        'label': '24-Hour QPE (in)',
    },
    'refl': {
        'prefix': 'CONUS/MergedBaseReflectivityQC_00.50',
        'bounds': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75],  # dBZ
        'colors': [
            '#646464',  # 0-5  dBZ : dark gray
            '#04e9e7',  # 5-10      : cyan
            '#019ff4',  # 10-15     : light blue
            '#0300f4',  # 15-20     : blue
            '#02fd02',  # 20-25     : bright green
            '#01c501',  # 25-30     : green
            '#008e00',  # 30-35     : dark green
            '#fdf802',  # 35-40     : yellow
            '#e5bc00',  # 40-45     : amber
            '#fd9500',  # 45-50     : orange
            '#fd0000',  # 50-55     : red
            '#d40000',  # 55-60     : dark red
            '#bc0000',  # 60-65     : darker red
            '#f800fd',  # 65-75     : magenta
        ],
        'min_val': 5.0,
        'max_val': 80.0,
        'label': 'Base Reflectivity (dBZ)',
    },
    'lightning': {
        'prefix': 'CONUS/LightningProbabilityNext60minGrid_scale_1',
        'bounds': [5, 10, 20, 30, 40, 50, 60, 75, 90],  # percent probability
        'colors': [
            '#ffffb2',  # 5-10%  : pale yellow
            '#fed976',  # 10-20% : light yellow-orange
            '#feb24c',  # 20-30% : orange
            '#fd8d3c',  # 30-40% : darker orange
            '#fc4e2a',  # 40-50% : red-orange
            '#e31a1c',  # 50-60% : red
            '#bd0026',  # 60-75% : dark red
            '#800026',  # 75-90% : deep red
        ],
        'min_val': 5.0,
        'max_val': 100.0,
        'label': '1-Hr CG Lightning Probability (%)',
    },
    'rotation': {
        'prefix': 'CONUS/MergedAzShear_0-2kmAGL_00.50',
        'bounds': [0.003, 0.005, 0.008, 0.012, 0.016, 0.020, 0.030, 0.040, 0.050],  # s^-1
        'colors': [
            '#00ff00',  # 0.003–0.005 s⁻¹ : bright green  (weak rotation)
            '#80ff00',  # 0.005–0.008 s⁻¹ : yellow-green
            '#ffff00',  # 0.008–0.012 s⁻¹ : yellow
            '#ff8000',  # 0.012–0.016 s⁻¹ : orange
            '#ff0000',  # 0.016–0.020 s⁻¹ : red
            '#c00000',  # 0.020–0.030 s⁻¹ : dark red      (strong rotation)
            '#ff00ff',  # 0.030–0.040 s⁻¹ : magenta
            '#8000ff',  # 0.040–0.050 s⁻¹ : purple        (tornadic)
        ],
        # Raw GRIB2 values are integers in units of 10⁻³ s⁻¹ (e.g. raw 5 = 0.005 s⁻¹).
        # Multiply by 0.001 to convert to s⁻¹ before applying bounds/min_val/max_val.
        'conversion': 0.001,
        # AzShear stores signed values (+ cyclonic, − anti-cyclonic).
        # use_abs=True renders both directions on the same colour scale.
        'use_abs': True,
        'min_val': 0.002,   # s⁻¹ — below this is noise
        'max_val': 1.0,     # s⁻¹ — generous cap; gross fills stripped earlier (>1e10)
        # AzShear couplets are only a few pixels across and scattered; dilate
        # before blurring so the smoothing doesn't erase them.
        'smooth_sigma': 1.2,
        'smooth_dilate': 2,
        'label': 'Azimuthal Shear (s\u207b\u00b9)',
    },
}

# --- HRRR PRECIP TYPE (model-based, avoids radar artefacts) ---
HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

_hrrr_cache = {}  # key: YYYYMMDDHH → (result_dict | None, reason_str | None)

def _fetch_hrrr_vars_s3(date_str, hour_str, hours_back):
    import cfgrib  # noqa: F401

    # CRAIN/CSNOW/CICEP/CFRZR are not present in fhr=00; use fhr>=1.
    # Using fhr=max(1,hours_back) keeps the valid time close to "now":
    #   hours_back=0 → fhr=01 (current run, valid ~1h ahead)
    #   hours_back=1 → fhr=01 (1h-old run, valid exactly now)
    #   hours_back=2 → fhr=02 (2h-old run, valid exactly now)
    fhr = max(1, hours_back)
    base_url = (
        f"{HRRR_BUCKET}/hrrr.{date_str}/conus"
        f"/hrrr.t{hour_str}z.wrfsfcf{fhr:02d}.grib2"
    )
    idx_url = base_url + ".idx"

    resp = session.get(idx_url, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"idx fetch failed: HTTP {resp.status_code}")

    target_vars = {'CRAIN', 'CSNOW', 'CICEP', 'CFRZR'}
    lines = resp.text.strip().split('\n')
    ranges = []
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) < 4:
            continue
        var_name = parts[3]
        if var_name in target_vars:
            start = int(parts[1])
            end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else ''
            ranges.append((var_name, start, end))

    if not ranges:
        raise ValueError("No CRAIN/CSNOW/CICEP/CFRZR found in .idx")

    fname = f"hrrr_cat_{hours_back}.grib2"
    with open(fname, 'wb') as f_out:
        for var_name, start, end in ranges:
            rng_header = f"bytes={start}-{end}" if end != '' else f"bytes={start}-"
            r = session.get(base_url, headers={'Range': rng_header}, timeout=30)
            if r.status_code not in (200, 206):
                raise ValueError(
                    f"Range request failed for {var_name}: HTTP {r.status_code}"
                )
            f_out.write(r.content)

    return fname


def get_hrrr_precip_type(target_dt, tgt_lats_2d, tgt_lons_2d):
    """Returns (result_dict, None) on success, or (None, reason_str) on failure."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, "scipy not installed"

    hrrr_hour_dt = target_dt.replace(minute=0, second=0, microsecond=0)
    cache_key = hrrr_hour_dt.strftime('%Y%m%d%H')
    if cache_key in _hrrr_cache:
        return _hrrr_cache[cache_key]

    attempt_errors = []
    for hours_back in range(3):
        run_dt   = hrrr_hour_dt - timedelta(hours=hours_back)
        date_str = run_dt.strftime('%Y%m%d')
        hour_str = run_dt.strftime('%H')
        fhr      = max(1, hours_back)
        label    = f"t-{hours_back}h ({hour_str}z fhr={fhr:02d})"

        fname  = None
        all_ds = []
        try:
            fname = _fetch_hrrr_vars_s3(date_str, hour_str, hours_back)

            import cfgrib
            all_ds = cfgrib.open_datasets(fname, backend_kwargs={'indexpath': ''})

            vmap = {}
            hrrr_lat = hrrr_lon = None
            for ds in all_ds:
                for vname in ds.data_vars:
                    vmap[vname.lower()] = ds[vname].values
                if hrrr_lat is None:
                    lat_k = next((k for k in ['latitude', 'lat'] if k in ds.coords), None)
                    lon_k = next((k for k in ['longitude', 'lon'] if k in ds.coords), None)
                    if lat_k and lon_k:
                        hrrr_lat = ds[lat_k].values
                        hrrr_lon = ds[lon_k].values

            if hrrr_lat is None or not vmap:
                msg = (f"{label}: cfgrib returned no usable datasets "
                       f"(vars found: {list(vmap)})")
                attempt_errors.append(msg)
                print(f"  HRRR {msg}")
                continue

            # HRRR GRIB2 stores longitudes in 0-360; normalize to -180/180
            # to match target_lons (LON_LEFT=-130, LON_RIGHT=-60).
            hrrr_lon = ((hrrr_lon + 180) % 360) - 180

            zero  = np.zeros_like(hrrr_lat)
            crain = vmap.get('crain', zero)
            csnow = vmap.get('csnow', zero)
            cicep = vmap.get('cicep', zero)
            cfrzr = vmap.get('cfrzr', zero)

            flat_lons = hrrr_lon.flatten()
            flat_lats = hrrr_lat.flatten()
            tree      = cKDTree(np.column_stack([flat_lons, flat_lats]))
            _, idxs   = tree.query(
                np.column_stack([tgt_lons_2d.flatten(), tgt_lats_2d.flatten()])
            )
            sh = tgt_lons_2d.shape

            def regrid(arr):
                return (arr.flatten()[idxs] >= 0.5).reshape(sh)

            result = {
                'rain': regrid(crain),
                'snow': regrid(csnow),
                'ice':  regrid(cicep) | regrid(cfrzr),
            }

            for ds in all_ds:
                ds.close()
            if fname and os.path.exists(fname):
                os.remove(fname)

            n_typed = int(np.sum(result['rain'] | result['snow'] | result['ice']))
            print(f"  HRRR precip type OK: {label} ({n_typed} typed px)")
            ret = (result, None)
            _hrrr_cache[cache_key] = ret
            return ret

        except Exception as e:
            msg = f"{label}: {e}"
            attempt_errors.append(msg)
            print(f"  HRRR attempt failed — {msg}")
            for ds in all_ds:
                try:
                    ds.close()
                except Exception:
                    pass
            if fname and os.path.exists(fname):
                os.remove(fname)

    reason = " | ".join(attempt_errors) if attempt_errors else "all 3 attempts failed (no detail)"
    ret = (None, reason)
    _hrrr_cache[cache_key] = ret
    return ret

# ---------------------------------------------------------------------------

def discover_rate_prefix():
    print("Finding current Rate prefix...")
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = session.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                p = element.text
                if "PrecipRate" in p or "SurfacePrecip" in p:
                    return p.rstrip("/")
    except:
        pass
    return "CONUS/SurfacePrecipRate_00.00"

def get_s3_keys(date_str, prefix):
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        return sorted([
            e.text for e in root.iter()
            if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')
        ])
    except:
        return []

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    print(f"  Downloading {key}...", end="", flush=True)
    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filename + ".gz", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(filename + ".gz")
        print(" ok")
        return True
    except Exception as e:
        print(f" Failed: {e}")
        return False

def _get_mercator_grid():
    """Return (width_px, height_px, target_lats, target_lons) for the standard grid."""
    res_scale = 100
    width_px  = int((LON_RIGHT - LON_LEFT) * res_scale)
    merc_top  = lat_to_merc(LAT_TOP)
    merc_bot  = lat_to_merc(LAT_BOT)
    merc_height_ratio = (merc_top - merc_bot) / np.radians(LON_RIGHT - LON_LEFT)
    height_px = int(width_px * merc_height_ratio)
    target_y    = np.linspace(merc_top, merc_bot, height_px)
    target_lats = merc_to_lat(target_y)
    target_lons = np.linspace(LON_LEFT, LON_RIGHT, width_px)
    return width_px, height_px, target_lats, target_lons

def _parse_valid_time(ds, fallback_key):
    """Extract valid_time from dataset, falling back to filename timestamp."""
    try:
        raw_time = ds.valid_time.values
        if isinstance(raw_time, np.ndarray):
            raw_time = raw_time.flat[0]
        return datetime.fromtimestamp(
            raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
    except Exception:
        ts_part = fallback_key.split('_')[-1]
        return datetime.strptime(
            ts_part.split('.')[0], "%Y%m%d-%H%M%S"
        ).replace(tzinfo=timezone.utc)

def process_frame(rate_key, flag_keys):
    """Process one precipitation rate frame and save as frame 0 (master.geojson)."""
    timestamp_part = rate_key.split('_')[-1]
    time_prefix    = timestamp_part[:13]
    flag_key       = next((k for k in flag_keys if time_prefix in k), None)

    tmp_r, tmp_f = "rate_0.grib2", None

    try:
        if not download_and_extract(rate_key, tmp_r):
            return

        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds_rate.coords['longitude'] = ((ds_rate.longitude + 180) % 360) - 180
        ds_rate = ds_rate.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        width_px, height_px, target_lats, target_lons = _get_mercator_grid()

        r_warp    = ds_rate[list(ds_rate.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest")
        rate_vals = r_warp.values

        utc_dt = _parse_valid_time(ds_rate, rate_key)
        tgt_lons_2d, tgt_lats_2d = np.meshgrid(target_lons, target_lats)

        hrrr, hrrr_reason = get_hrrr_precip_type(utc_dt, tgt_lats_2d, tgt_lons_2d)

        if hrrr is not None:
            has_precip    = rate_vals > 0.1
            rain_mask     = hrrr['rain'] & has_precip
            snow_mask     = hrrr['snow'] & has_precip & ~hrrr['rain']
            ice_mask      = hrrr['ice']  & has_precip & ~hrrr['rain'] & ~hrrr['snow']
            still_untyped = has_precip & ~(rain_mask | snow_mask | ice_mask)
            rain_mask    |= still_untyped

        else:
            print(f"  HRRR failed — reason: {hrrr_reason}")
            if not flag_key:
                print(f"  Skipping {time_prefix}: no HRRR and no matching PrecipFlag file.")
                return

            print(f"  Falling back to MRMS PrecipFlag.")
            tmp_f = "flag_0.grib2"
            if not download_and_extract(flag_key, tmp_f):
                return

            ds_flag = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath': ''})
            ds_flag.coords['longitude'] = ((ds_flag.longitude + 180) % 360) - 180
            ds_flag = ds_flag.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
            f_warp    = ds_flag[list(ds_flag.data_vars)[0]].interp(
                latitude=target_lats, longitude=target_lons, method="nearest")
            flag_vals = f_warp.values
            ds_flag.close()

            has_precip        = rate_vals > 0.1
            rain_mask         = np.isin(flag_vals, [1, 2, 91, 96])
            snow_mask         = np.isin(flag_vals, [3])
            ice_mask          = np.isin(flag_vals, [4, 5, 6])
            untyped_with_rate = has_precip & np.isin(flag_vals, [7, 10])
            rain_mask        |= untyped_with_rate

        high_rate  = rate_vals > WINTRY_RATE_MAX  # threshold in mm/hr (raw data)
        rain_mask |= (snow_mask | ice_mask) & high_rate
        snow_mask &= ~high_rate
        ice_mask  &= ~high_rate

        # Convert mm/hr → in/hr for display; RAIN/SNOW/ICE_BOUNDS are in inches
        rain = np.where(rain_mask, rate_vals * MM_TO_IN, np.nan)
        snow = np.where(snow_mask, rate_vals * MM_TO_IN, np.nan)
        ice  = np.where(ice_mask,  rate_vals * MM_TO_IN, np.nan)

        # Anything the rate grid called precipitation (>0.1 mm/hr) should show,
        # including rates just under the first colour bound.
        rate_floor = 0.1 * MM_TO_IN  # in/hr

        # Each precip type is contoured separately so the three colour tables
        # stay independent. The masks are mutually exclusive, so the bands
        # don't overlap; rain is emitted first, snow and ice paint over it
        # wherever their smoothing halos meet.
        features = []
        for arr, arr_bounds, arr_colors in ((rain, RAIN_BOUNDS, RAIN_COLORS),
                                            (snow, SNOW_BOUNDS, SNOW_COLORS),
                                            (ice,  ICE_BOUNDS,  ICE_COLORS)):
            if np.any(arr > arr_bounds[0]):
                features += contour_features(arr, arr_bounds, arr_colors,
                                             width_px, height_px,
                                             min_val=rate_floor)

        geo_path = os.path.join(OUTPUT_DIR, "master.geojson")
        nbytes   = save_geojson(features, geo_path)
        print(f"  Precip rate contours: {len(features)} polygons, "
              f"{nbytes/1e6:.2f} MB")

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta  = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time":   et_dt.strftime("%I:%M %p ET"),
        }
        with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
            json.dump(meta, f)

        print(f"  Precip rate frame 0 saved: {meta['time']} ({width_px}x{height_px})")

        ds_rate.close()
        gc.collect()

    except Exception as e:
        print(f"  Error on precip rate frame: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if f and os.path.exists(f):
                os.remove(f)


def _open_grib2_dataset(tmp_file):
    """Open a GRIB2 file with cfgrib, falling back to cfgrib.open_datasets for
    non-standard MRMS products that xr.open_dataset can't handle as a single message."""
    try:
        ds = xr.open_dataset(tmp_file, engine="cfgrib", backend_kwargs={'indexpath': ''})
        if len(ds.data_vars) == 0:
            raise ValueError("empty dataset from xr.open_dataset")
        return ds, None
    except Exception as e_single:
        # Some MRMS products (e.g. RotationTrack) have GRIB2 messages that cfgrib
        # can't aggregate into one dataset — try open_datasets and merge the first.
        try:
            import cfgrib
            all_ds = cfgrib.open_datasets(tmp_file, backend_kwargs={'indexpath': ''})
            if not all_ds:
                raise ValueError("cfgrib.open_datasets returned no datasets")
            # Pick the dataset that has data variables
            ds = next((d for d in all_ds if len(d.data_vars) > 0), None)
            if ds is None:
                raise ValueError("no data vars in any cfgrib dataset")
            # Close the ones we're not using
            for d in all_ds:
                if d is not ds:
                    try:
                        d.close()
                    except Exception:
                        pass
            return ds, all_ds
        except Exception as e_multi:
            raise RuntimeError(
                f"cfgrib single-dataset failed ({e_single}); "
                f"multi-dataset fallback also failed ({e_multi})"
            )


_INSPECTABLE_PRODUCTS = {'mesh', 'qpe6h', 'qpe24h', 'lightning', 'rotation'}


def _save_value_png(vals, max_val, filepath):
    """Encode a float array as a value PNG for exact browser pixel inspection.

    Encoding: R = high byte, G = low byte of uint16(val / max_val * 65535).
    Alpha = 255 for valid (finite, >0) pixels, 0 for NaN/missing.
    Precision ≈ max_val / 65535.
    """
    valid = np.isfinite(vals) & (vals > 0)
    h, w = vals.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if np.any(valid):
        scaled = np.clip(vals[valid] / max_val * 65535, 0, 65535).astype(np.uint16)
        rgba[valid, 0] = (scaled >> 8).astype(np.uint8)    # high byte → R
        rgba[valid, 1] = (scaled & 0xFF).astype(np.uint8)  # low byte  → G
        rgba[valid, 3] = 255                                # opaque
    plt.imsave(filepath, rgba)


def process_single_field_frame(key, product_key, product_cfg):
    """Download, process, and render one frame of a single-field MRMS product.

    Saves output as {product_key}_0.geojson and metadata_{product_key}_0.json.
    For inspectable products also saves {product_key}_val_0.png with exact
    float values encoded in R+G channels for browser pixel inspection.
    The GitHub Actions workflow shifts these to _1, _2, ... before each run,
    maintaining a rolling NUM_FRAMES archive.
    """
    tmp_file = f"tmp_{product_key}.grib2"
    all_ds_refs = None  # track extra cfgrib dataset refs for cleanup

    try:
        if not download_and_extract(key, tmp_file):
            return

        ds, all_ds_refs = _open_grib2_dataset(tmp_file)
        ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        ds = ds.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        width_px, height_px, target_lats, target_lons = _get_mercator_grid()

        warped = ds[list(ds.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest"
        )
        vals = warped.values.astype(float)

        # Diagnostic: show raw data range before any processing
        raw_finite = vals[np.isfinite(vals)]
        if raw_finite.size > 0:
            print(f"  {product_key} raw: min={raw_finite.min():.4g}, "
                  f"max={raw_finite.max():.4g}, "
                  f"non-fill pixels={raw_finite.size}")
        else:
            print(f"  {product_key} raw: all fill/NaN — file may be empty")

        # Mask obvious GRIB2 fill values (±9.99e+20 and similar) before abs/conversion
        vals[np.abs(vals) > 1e10] = np.nan

        # For signed products (e.g. rotation track which stores both cyclonic +
        # and anti-cyclonic − azimuthal shear), take the absolute value so that
        # both rotation directions are rendered.
        if product_cfg.get('use_abs', False):
            vals = np.abs(vals)

        # Apply unit conversion if specified (e.g. mm → inches for precip products)
        conversion = product_cfg.get('conversion', 1.0)
        if conversion != 1.0:
            vals = vals * conversion

        # Mask fill values and below-threshold values.
        # min_val/max_val are in the same units as the (optionally converted) vals.
        min_val = product_cfg['min_val']
        max_val = product_cfg['max_val']
        vals[(vals < min_val) | (vals > max_val)] = np.nan

        n_valid = int(np.sum(~np.isnan(vals)))
        print(f"  {product_key} after filtering [{min_val}, {max_val}]: "
              f"{n_valid} valid pixels")

        utc_dt = _parse_valid_time(ds, key)

        features = []
        if n_valid > 0:
            features = contour_features(
                vals, product_cfg['bounds'], product_cfg['colors'],
                width_px, height_px,
                sigma=product_cfg.get('smooth_sigma', SMOOTH_SIGMA),
                dilate=product_cfg.get('smooth_dilate', SMOOTH_DILATE),
                min_val=min_val,
            )

        geo_path  = os.path.join(OUTPUT_DIR, f"{product_key}_0.geojson")
        meta_path = os.path.join(OUTPUT_DIR, f"metadata_{product_key}_0.json")

        nbytes = save_geojson(features, geo_path)
        print(f"  {product_key} contours: {len(features)} polygons, "
              f"{nbytes/1e6:.2f} MB")

        if product_key in _INSPECTABLE_PRODUCTS and n_valid > 0:
            val_path = os.path.join(OUTPUT_DIR, f"{product_key}_val_0.png")
            _save_value_png(vals, max_val, val_path)
            print(f"  {product_key} value PNG saved → {val_path}")

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta  = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time":   et_dt.strftime("%I:%M %p ET"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        print(f"  {product_key} frame 0 saved: {meta['time']} ({width_px}x{height_px})")

        ds.close()
        gc.collect()

    except Exception as e:
        print(f"  Error processing {product_key}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if all_ds_refs:
            for d in all_ds_refs:
                try:
                    d.close()
                except Exception:
                    pass
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)

    # --- Precipitation Rate: process only the single latest frame ---
    print("=== Precipitation Rate ===")
    precip_done = False
    for d in range(2):
        date_str  = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        if not rate_keys:
            print(f"  No rate keys for {date_str}.")
            continue
        latest_rate_key = sorted(rate_keys)[-1]
        print(f"  Latest key: {latest_rate_key}")
        process_frame(latest_rate_key, flag_keys)
        precip_done = True
        break
    if not precip_done:
        print("  No precipitation rate data found.")

    # --- Single-field products: process only the single latest frame each ---
    for product_key, product_cfg in SINGLE_PRODUCTS.items():
        print(f"\n=== {product_cfg['label']} ===")
        product_done = False
        for d in range(2):
            date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
            keys     = get_s3_keys(date_str, product_cfg['prefix'])
            if not keys:
                print(f"  No keys for {date_str}.")
                continue
            latest_key = sorted(keys)[-1]
            print(f"  Latest key: {latest_key}")
            process_single_field_frame(latest_key, product_key, product_cfg)
            product_done = True
            break
        if not product_done:
            print(f"  No data found for {product_key}.")
