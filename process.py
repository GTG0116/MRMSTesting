"""
process.py - GRIB2 decoding, regridding, smoothing, and image rendering.
"""
import logging
import os
import tempfile
import warnings
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image

try:
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False

try:
    import eccodes
    HAS_ECCODES = True
except ImportError:
    HAS_ECCODES = False

logger = logging.getLogger(__name__)

# MRMS CONUS native grid
MRMS_NROWS = 3500
MRMS_NCOLS = 7000
MRMS_NW_LAT = 54.995
MRMS_NW_LON = -129.995  # -180..180
MRMS_SE_LAT = 20.005
MRMS_SE_LON = -60.005
MRMS_DY = -0.01   # lat decreases row-by-row (N→S)
MRMS_DX = 0.01    # lon increases col-by-col (W→E)

# Downsample factor (8 → 875×437 output image)
DOWNSAMPLE = 8

# Gaussian sigma applied to the downsampled rate array
# sigma=1.5 at DOWNSAMPLE=8 ≈ 12 km smoothing, similar to nowCOAST
SMOOTH_SIGMA = 1.5

# Precipitation type codes
PTYPE_NONE = 0
PTYPE_RAIN = 1
PTYPE_MIX = 2
PTYPE_SNOW = 3
PTYPE_FZRA = 7
PTYPE_IP = 8
PTYPE_HAIL = 91  # MRMS PrecipFlag hail code

# Rate thresholds for display (mm/hr)
MIN_RATE = 0.1


# ---------------------------------------------------------------------------
# Color tables  (NWS QPE style, nowCOAST-inspired)
# ---------------------------------------------------------------------------

# (min_rate, R, G, B)  — exclusive lower bound, inclusive upper
_RAIN_COLORS = [
    (0.10,  (4,   233, 231)),   # trace – teal
    (0.50,  (1,   159, 244)),   # light – sky blue
    (1.00,  (3,   0,   244)),   # light mod – blue
    (2.00,  (2,   253, 2)),     # mod – bright green
    (4.00,  (1,   197, 1)),     # mod+ – green
    (6.00,  (0,   142, 0)),     # mod-heavy – dark green
    (10.0,  (253, 248, 2)),     # heavy – yellow
    (15.0,  (229, 188, 0)),     # heavy+ – gold
    (20.0,  (253, 149, 0)),     # very heavy – orange
    (30.0,  (253, 0,   0)),     # extreme – red
    (45.0,  (212, 0,   0)),     # extreme+ – dark red
    (60.0,  (188, 0,   253)),   # hail/extreme – purple
]

_SNOW_COLORS = [
    (0.10,  (189, 225, 255)),
    (0.50,  (100, 196, 255)),
    (1.00,  (0,   144, 255)),
    (2.00,  (0,   64,  210)),
    (4.00,  (0,   0,   168)),
]

_FZRA_COLORS = [
    (0.10,  (255, 183, 255)),
    (1.00,  (255, 0,   255)),
    (5.00,  (178, 0,   178)),
]

_IP_COLORS = [
    (0.10,  (210, 180, 255)),
    (1.00,  (140, 0,   255)),
]

_MIX_COLORS = [
    (0.10,  (255, 210, 100)),
    (2.00,  (210, 140, 0)),
    (5.00,  (160, 100, 0)),
]


def _lookup_color(rate: np.ndarray, table) -> np.ndarray:
    """Vectorized color lookup.  Returns (H,W,3) uint8 array."""
    h, w = rate.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    thresholds = [t[0] for t in table]
    colors = [t[1] for t in table]

    for i, (thresh, color) in enumerate(zip(thresholds, colors)):
        next_thresh = thresholds[i + 1] if i + 1 < len(thresholds) else 1e9
        mask = (rate >= thresh) & (rate < next_thresh)
        rgb[mask] = color

    # anything >= last threshold
    last_thresh, last_color = table[-1]
    rgb[rate >= last_thresh] = last_color

    return rgb


def _smooth_alpha(rate: np.ndarray) -> np.ndarray:
    """Compute alpha channel: smooth ramp from transparent→opaque at the
    precipitation edge, then constant opaque.  Prevents hard cutoffs."""
    alpha = np.zeros(rate.shape, dtype=np.uint8)
    ramp_low, ramp_high = MIN_RATE, MIN_RATE * 3.0
    ramp = np.clip((rate - ramp_low) / (ramp_high - ramp_low), 0, 1)
    alpha = (ramp * 210).astype(np.uint8)
    return alpha


# ---------------------------------------------------------------------------
# GRIB2 decoding
# ---------------------------------------------------------------------------

def _decode_with_cfgrib(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode using cfgrib (requires xarray). Returns (values, lats_1d_or_2d, lons_1d_or_2d)."""
    if not HAS_CFGRIB:
        return None
    try:
        fn = getattr(cfgrib, "open_datasets", None)
        if fn is None:
            return None
        datasets = fn(filepath, errors="ignore", indexpath=None)
    except Exception as e:
        logger.warning("cfgrib open failed: %s", e)
        return None

    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            values = da.values.astype(np.float32)
            if values.ndim < 2:
                continue

            # Resolve coordinate arrays
            lat_key = next((k for k in ("latitude", "y") if k in ds.coords), None)
            lon_key = next((k for k in ("longitude", "x") if k in ds.coords), None)
            if lat_key is None or lon_key is None:
                continue

            lats = ds[lat_key].values.astype(np.float32)
            lons = ds[lon_key].values.astype(np.float32)

            # Mask fill values
            fill = float(da.attrs.get("GRIB_missingValue", -9999.0))
            values[np.abs(values - fill) < 1.0] = np.nan
            values[values < -900] = np.nan

            logger.info(
                "cfgrib decoded '%s' shape=%s lat=[%.2f,%.2f] lon=[%.2f,%.2f]",
                var, values.shape,
                float(np.nanmin(lats)), float(np.nanmax(lats)),
                float(np.nanmin(lons)), float(np.nanmax(lons)),
            )
            return values, lats, lons

    return None


def _decode_with_eccodes(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Low-level eccodes fallback decoder."""
    if not HAS_ECCODES:
        return None
    try:
        with open(filepath, "rb") as f:
            handle = eccodes.codes_grib_new_from_file(f)
            if handle is None:
                return None
            try:
                ni = eccodes.codes_get(handle, "Ni")
                nj = eccodes.codes_get(handle, "Nj")
                values = eccodes.codes_get_values(handle).astype(np.float32)
                lats = eccodes.codes_get_array(handle, "latitudes").astype(np.float32)
                lons = eccodes.codes_get_array(handle, "longitudes").astype(np.float32)
                missing = float(eccodes.codes_get(handle, "missingValue"))
                values[np.abs(values - missing) < 1.0] = np.nan
                values[values < -900] = np.nan
                values = values.reshape(nj, ni)
                lats = lats.reshape(nj, ni)
                lons = lons.reshape(nj, ni)
                logger.info("eccodes decoded shape=%s", values.shape)
                return values, lats, lons
            finally:
                eccodes.codes_release(handle)
    except Exception as e:
        logger.warning("eccodes decode failed: %s", e)
        return None


def decode_grib2_bytes(grib_bytes: bytes) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode raw GRIB2 bytes (full file or single record) to (values, lats, lons)."""
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        path = f.name
    try:
        result = _decode_with_cfgrib(path)
        if result is None:
            result = _decode_with_eccodes(path)
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Precip type helpers
# ---------------------------------------------------------------------------

def build_ptype_from_categorical(
    crain: Optional[np.ndarray],
    csnow: Optional[np.ndarray],
    cfrzr: Optional[np.ndarray],
    cicep: Optional[np.ndarray],
) -> np.ndarray:
    """
    Combine four binary (0/1) categorical arrays into one integer type array.
    Priority: rain > snow > ice-pellets > freezing-rain > none.
    """
    if all(x is None for x in (crain, csnow, cfrzr, cicep)):
        return None

    shape = next(a.shape for a in (crain, csnow, cfrzr, cicep) if a is not None)
    ptype = np.zeros(shape, dtype=np.uint8)

    if cfrzr is not None:
        ptype[cfrzr > 0.5] = PTYPE_FZRA
    if cicep is not None:
        ptype[cicep > 0.5] = PTYPE_IP
    if csnow is not None:
        ptype[csnow > 0.5] = PTYPE_SNOW
    if crain is not None:
        ptype[crain > 0.5] = PTYPE_RAIN

    return ptype


def _normalize_lons(lons: np.ndarray) -> np.ndarray:
    """Convert 0-360 longitudes to -180..180."""
    return np.where(lons > 180, lons - 360, lons)


def regrid_nearest(
    src_values: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    dst_lats_1d: np.ndarray,
    dst_lons_1d: np.ndarray,
) -> np.ndarray:
    """
    Nearest-neighbour regrid from source (possibly 2-D irregular) lat/lon grid
    to a destination regular lat/lon grid.
    """
    src_lons = _normalize_lons(src_lons)

    # Flatten source coordinates
    if src_lats.ndim == 2:
        flat_lats = src_lats.ravel()
        flat_lons = src_lons.ravel()
        flat_vals = src_values.ravel()
    else:
        # 1-D regular grid → meshgrid
        lon2d, lat2d = np.meshgrid(src_lons, src_lats)
        flat_lats = lat2d.ravel()
        flat_lons = lon2d.ravel()
        flat_vals = src_values.ravel()

    # Build KDTree (cos-weighted for longitude near poles)
    cos_lat = np.cos(np.deg2rad(flat_lats))
    src_xy = np.column_stack([flat_lats, flat_lons * cos_lat])

    # Destination grid
    lon2d_dst, lat2d_dst = np.meshgrid(dst_lons_1d, dst_lats_1d)
    cos_lat_dst = np.cos(np.deg2rad(lat2d_dst))
    dst_xy = np.column_stack([
        lat2d_dst.ravel(),
        lon2d_dst.ravel() * cos_lat_dst.ravel(),
    ])

    tree = cKDTree(src_xy)
    _, idx = tree.query(dst_xy, workers=-1)

    result = flat_vals[idx].reshape(len(dst_lats_1d), len(dst_lons_1d))
    logger.info("Regridded %s → %s", src_values.shape, result.shape)
    return result


# ---------------------------------------------------------------------------
# Downsampling & smoothing
# ---------------------------------------------------------------------------

def downsample_array(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample, ignoring NaN."""
    h, w = arr.shape
    nh = h // factor
    nw = w // factor
    block = arr[: nh * factor, : nw * factor].reshape(nh, factor, nw, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(block, axis=(1, 3))
    return result


def smooth_rate(rate: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Apply Gaussian smoothing; treat NaN as 0 for convolution then re-mask."""
    nan_mask = np.isnan(rate)
    filled = np.where(nan_mask, 0.0, rate)
    weight = np.where(nan_mask, 0.0, 1.0)
    s_filled = gaussian_filter(filled.astype(np.float32), sigma=sigma)
    s_weight = gaussian_filter(weight.astype(np.float32), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(s_weight > 0, s_filled / s_weight, np.nan)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# MRMS native lat/lon grids
# ---------------------------------------------------------------------------

def mrms_lats_1d() -> np.ndarray:
    """1-D latitude array for the MRMS CONUS native grid (N→S)."""
    return np.linspace(MRMS_NW_LAT, MRMS_SE_LAT, MRMS_NROWS).astype(np.float32)


def mrms_lons_1d() -> np.ndarray:
    """1-D longitude array for the MRMS CONUS native grid (W→E), -180..180."""
    return np.linspace(MRMS_NW_LON, MRMS_SE_LON, MRMS_NCOLS).astype(np.float32)


def downsampled_lats_lons() -> Tuple[np.ndarray, np.ndarray]:
    """Lat/lon 1-D arrays for the downsampled output grid."""
    h = MRMS_NROWS // DOWNSAMPLE
    w = MRMS_NCOLS // DOWNSAMPLE
    lats = np.linspace(MRMS_NW_LAT, MRMS_SE_LAT, h).astype(np.float32)
    lons = np.linspace(MRMS_NW_LON, MRMS_SE_LON, w).astype(np.float32)
    return lats, lons


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_rgba_image(smoothed_rate: np.ndarray, ptype: np.ndarray) -> np.ndarray:
    """
    Build an RGBA uint8 numpy array from smoothed precip rate and type.
    ptype values: 0=none, 1=rain, 2=mix, 3=snow, 7=fzra, 8=ip.
    """
    h, w = smoothed_rate.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    rate_safe = np.nan_to_num(smoothed_rate, nan=0.0)
    alpha = _smooth_alpha(rate_safe)

    # Build masks (only where there's meaningful precip)
    precip_mask = rate_safe >= MIN_RATE

    rain_m = precip_mask & ((ptype == PTYPE_RAIN) | (ptype == PTYPE_HAIL))
    snow_m = precip_mask & (ptype == PTYPE_SNOW)
    fzra_m = precip_mask & ((ptype == PTYPE_FZRA) | (ptype == 10))
    ip_m   = precip_mask & (ptype == PTYPE_IP)
    mix_m  = precip_mask & ((ptype == PTYPE_MIX) | (ptype == 6))
    # Anything with precip but unknown type → treat as rain
    other_m = precip_mask & ~(rain_m | snow_m | fzra_m | ip_m | mix_m)

    for mask, table in [
        (rain_m | other_m, _RAIN_COLORS),
        (snow_m,           _SNOW_COLORS),
        (fzra_m,           _FZRA_COLORS),
        (ip_m,             _IP_COLORS),
        (mix_m,            _MIX_COLORS),
    ]:
        if not np.any(mask):
            continue
        rgb = _lookup_color(rate_safe, table)
        rgba[mask, :3] = rgb[mask]
        rgba[mask, 3] = alpha[mask]

    return rgba


def generate_frame(
    rate_bytes: bytes,
    ptype_cat: Optional[Dict[str, bytes]],
    mrms_ptype_bytes: Optional[bytes],
    output_path: str,
) -> dict:
    """
    Full pipeline:
      1. Decode MRMS PrecipRate
      2. Decode and regrid precip type (NWP categorical → MRMS → fallback)
      3. Downsample rate & type
      4. Smooth rate
      5. Render RGBA image
      6. Save PNG (no browser-side anti-aliasing: image is already smooth)
    Returns metadata dict for the viewer.
    """
    # --- 1. Decode MRMS PrecipRate ---
    logger.info("Decoding MRMS PrecipRate ...")
    rate_result = decode_grib2_bytes(rate_bytes)
    if rate_result is None:
        raise RuntimeError("Failed to decode MRMS PrecipRate grib2")
    rate_arr, rate_lats, rate_lons = rate_result

    # Orient N→S, W→E
    if rate_lats.ndim == 1:
        if rate_lats[0] < rate_lats[-1]:          # ascending → flip
            rate_arr = np.flipud(rate_arr)
            rate_lats = rate_lats[::-1]
        rate_lons_norm = _normalize_lons(rate_lons)
        if rate_lons_norm[0] > rate_lons_norm[-1]:  # descending lons → flip
            rate_arr = np.fliplr(rate_arr)
            rate_lons_norm = rate_lons_norm[::-1]
    else:
        rate_lons_norm = _normalize_lons(rate_lons)

    # Clamp negative rates to NaN
    rate_arr[rate_arr < 0] = np.nan
    logger.info("Rate array shape: %s, max=%.2f", rate_arr.shape, float(np.nanmax(rate_arr)))

    # --- 2. Decode precip type ---
    ptype_full = None
    ptype_source_used = "unknown"

    # Try NWP categorical (from HRRR or RAP)
    if ptype_cat:
        logger.info("Decoding NWP categorical precip type ...")
        cat_arrays: Dict[str, Optional[np.ndarray]] = {}
        cat_lats = cat_lons = None

        for var, data in ptype_cat.items():
            result = decode_grib2_bytes(data)
            if result is None:
                continue
            vals, lts, lns = result
            if vals.ndim < 2:
                continue
            cat_arrays[var] = vals
            cat_lats, cat_lns_raw = lts, lns

        if cat_arrays and cat_lats is not None:
            cat_lons_norm = _normalize_lons(cat_lns_raw)
            ptype_src = build_ptype_from_categorical(
                cat_arrays.get("CRAIN"),
                cat_arrays.get("CSNOW"),
                cat_arrays.get("CFRZR"),
                cat_arrays.get("CICEP"),
            )
            if ptype_src is not None:
                logger.info("Regridding NWP precip type to MRMS grid ...")
                native_lats, native_lons = mrms_lats_1d(), mrms_lons_1d()
                ptype_full = regrid_nearest(
                    ptype_src.astype(np.float32),
                    cat_lats,
                    cat_lons_norm,
                    native_lats,
                    native_lons,
                ).astype(np.uint8)
                logger.info("NWP ptype regridded to %s", ptype_full.shape)
                # If the NWP model gave all-zero categorical precip (common at
                # f=00 analysis time before spin-up), discard and try MRMS.
                if not np.any(ptype_full > 0):
                    logger.info("NWP ptype is all zeros; discarding for MRMS fallback")
                    ptype_full = None
                else:
                    ptype_source_used = "HRRR/RAP"

    # Fall back to MRMS PrecipType/PrecipFlag
    if ptype_full is None and mrms_ptype_bytes is not None:
        logger.info("Using MRMS fallback precip type ...")
        pt_result = decode_grib2_bytes(mrms_ptype_bytes)
        if pt_result is not None:
            pt_arr, pt_lats, pt_lons = pt_result
            # Clamp negative sentinel values (-3 = no coverage) to 0 before uint8 cast
            pt_arr = np.where(pt_arr < 0, 0, pt_arr)
            pt_arr = np.nan_to_num(pt_arr, nan=0)
            # Orient N→S
            if pt_lats.ndim == 1 and pt_lats[0] < pt_lats[-1]:
                pt_arr = np.flipud(pt_arr)
            # MRMS PrecipFlag codes:
            #   0=no precip, 1=rain, 3=snow, 6=mixed-warm, 7=fzra, 8=ip, 10=fzra-alt, 91=hail
            # Our generate_rgba_image already handles 1,2/6,3,7/10,8,91 so just cast.
            ptype_full = pt_arr.astype(np.uint8)
            ptype_source_used = "MRMS-PrecipFlag"
            logger.info("MRMS fallback ptype shape: %s", ptype_full.shape)

    # Last resort: assume everything is rain
    if ptype_full is None:
        logger.warning("No precip type available; assuming rain everywhere")
        ptype_full = np.ones(rate_arr.shape, dtype=np.uint8)
        ptype_source_used = "assumed-rain"

    # --- 3. Downsample ---
    logger.info("Downsampling by %dx ...", DOWNSAMPLE)
    rate_ds = downsample_array(rate_arr, DOWNSAMPLE)

    # For ptype, use mode (nearest): just take every Nth pixel (since it's categorical)
    ptype_ds = ptype_full[::DOWNSAMPLE, ::DOWNSAMPLE]

    # Ensure same shape
    h = min(rate_ds.shape[0], ptype_ds.shape[0])
    w = min(rate_ds.shape[1], ptype_ds.shape[1])
    rate_ds = rate_ds[:h, :w]
    ptype_ds = ptype_ds[:h, :w]

    # --- 4. Smooth rate ---
    logger.info("Smoothing rate (sigma=%.1f) ...", SMOOTH_SIGMA)
    rate_smooth = smooth_rate(rate_ds, sigma=SMOOTH_SIGMA)

    # --- 5. Render RGBA ---
    logger.info("Rendering RGBA image (%dx%d) ...", w, h)
    rgba = generate_rgba_image(rate_smooth, ptype_ds)

    # --- 6. Save PNG ---
    img = Image.fromarray(rgba, "RGBA")
    img.save(output_path, "PNG", optimize=False)
    logger.info("Saved frame to %s (%d×%d)", output_path, w, h)

    return {
        "width": w,
        "height": h,
        "bounds": [
            [float(MRMS_SE_LAT), float(MRMS_NW_LON)],   # SW
            [float(MRMS_NW_LAT), float(MRMS_SE_LON)],   # NE
        ],
        "max_rate": float(np.nanmax(rate_ds)),
        "mean_rate": float(np.nanmean(rate_ds[rate_ds > 0])) if np.any(rate_ds > 0) else 0.0,
        "ptype_source_used": ptype_source_used,
    }
