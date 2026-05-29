"""
fetch.py - Data fetching for MRMS viewer.
Downloads MRMS grib2 from S3, and categorical precip type from HRRR → RAP → MRMS fallback.
"""
import gzip
import io
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

import requests

logger = logging.getLogger(__name__)

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com"
HRRR_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/v4.1"
RAP_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/v5.1"

# S3 directory names for MRMS products (format: ProductName_threshold)
MRMS_S3_DIRS = {
    "PrecipRate": "PrecipRate_00.00",
    "PrecipFlag": "PrecipFlag_00.00",
    "PrecipType": "PrecipType_00.00",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MRMSViewer/1.0 (educational/research)"})


def _get(url: str, timeout: int = 30, **kwargs) -> requests.Response:
    """GET with simple retry on network errors."""
    delays = [2, 4, 8]
    for i, delay in enumerate(delays):
        try:
            r = SESSION.get(url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == len(delays) - 1:
                raise
            logger.warning("Fetch error %s, retrying in %ds: %s", e, delay, url)
            time.sleep(delay)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# MRMS S3 helpers
# ---------------------------------------------------------------------------

def _list_s3_prefix(prefix: str, max_keys: int = 10) -> list[str]:
    """List keys in the MRMS S3 bucket under prefix, sorted ascending."""
    url = (
        f"{MRMS_S3_BASE}?list-type=2"
        f"&prefix={requests.utils.quote(prefix, safe='/')}"
        f"&max-keys={max_keys}"
    )
    try:
        r = _get(url, timeout=20)
    except requests.RequestException as e:
        logger.debug("S3 list failed for prefix %s: %s", prefix, e)
        return []

    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [c.find("s3:Key", ns).text for c in root.findall("s3:Contents", ns)]
    return sorted(keys)


def find_latest_mrms_file(product: str) -> Optional[str]:
    """
    Return the full download URL for the most recent MRMS CONUS file for the
    given product (e.g. 'PrecipRate', 'PrecipFlag').

    S3 layout:  CONUS/{ProductName_threshold}/{YYYYMMDD}/MRMS_…_YYYYMMDD-HHMMSS.grib2.gz
    Tries today then yesterday to handle UTC midnight boundary.
    """
    dir_name = MRMS_S3_DIRS.get(product)
    if dir_name is None:
        logger.warning("Unknown MRMS product: %s", product)
        return None

    now_utc = datetime.now(timezone.utc)
    for days_back in range(2):
        dt = now_utc - timedelta(days=days_back)
        date_str = dt.strftime("%Y%m%d")
        prefix = f"CONUS/{dir_name}/{date_str}/"
        keys = _list_s3_prefix(prefix, max_keys=20)
        if keys:
            latest_key = keys[-1]   # alphabetically last = most recent 2-min timestamp
            url = f"{MRMS_S3_BASE}/{latest_key}"
            logger.info("Found latest %s: %s", product, url)
            return url

    logger.warning("No MRMS %s file found in S3", product)
    return None


def download_grib2(url: str) -> Optional[bytes]:
    """Download a .grib2.gz (or .grib2) and return decompressed bytes."""
    logger.info("Downloading %s", url)
    try:
        r = _get(url, timeout=120, stream=True)
        data = r.content
    except requests.RequestException as e:
        logger.error("Download failed: %s", e)
        return None

    if url.endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except Exception as e:
            logger.error("Decompression failed: %s", e)
            return None

    logger.info("Downloaded %d bytes (decompressed)", len(data))
    return data


# ---------------------------------------------------------------------------
# HRRR / RAP idx-based byte-range fetching
# ---------------------------------------------------------------------------

def _parse_idx(idx_text: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse a NOMADS .idx file into {var:level: (start_byte, end_byte)}.
    end_byte=-1 means 'to EOF'.
    """
    records: Dict[str, Tuple[int, int]] = {}
    lines = [l for l in idx_text.strip().splitlines() if l.strip()]

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            start = int(parts[1])
        except ValueError:
            continue
        var = parts[3].strip()
        level = parts[4].strip()
        key = f"{var}:{level}"

        # Determine end byte from next record's start
        end = -1
        for j in range(i + 1, len(lines)):
            nparts = lines[j].split(":")
            if len(nparts) >= 2:
                try:
                    end = int(nparts[1]) - 1
                    break
                except ValueError:
                    continue

        records[key] = (start, end)

    return records


def _fetch_grib_range(grib_url: str, var: str, level_contains: str) -> Optional[bytes]:
    """
    Download a single GRIB2 record from grib_url matching var at a level
    string that contains level_contains (case-insensitive), using the .idx.
    """
    idx_url = grib_url + ".idx"
    try:
        idx_resp = _get(idx_url, timeout=20)
    except requests.RequestException as e:
        logger.warning("idx fetch failed %s: %s", idx_url, e)
        return None

    records = _parse_idx(idx_resp.text)

    # Find matching key
    match_key = None
    for key in records:
        kvar, klevel = key.split(":", 1)
        if kvar.upper() == var.upper() and level_contains.lower() in klevel.lower():
            match_key = key
            break

    if match_key is None:
        logger.debug("Var %s not found in %s", var, idx_url)
        return None

    start, end = records[match_key]
    range_header = f"bytes={start}-{end}" if end != -1 else f"bytes={start}-"

    try:
        r = _get(grib_url, timeout=60, headers={"Range": range_header})
        logger.debug("Fetched %s %s: %d bytes", var, match_key, len(r.content))
        return r.content
    except requests.RequestException as e:
        logger.warning("Range fetch failed: %s", e)
        return None


def _try_nwp_run(base_url: str, filename_fn, dt: datetime) -> Optional[Dict[str, bytes]]:
    """
    Try to fetch CRAIN/CSNOW/CFRZR/CICEP records from a model run at dt.
    Returns dict of {varname: grib_bytes} or None.
    """
    grib_url = filename_fn(dt)
    logger.info("Trying NWP precip type from %s", grib_url)

    results = {}
    for var in ("CRAIN", "CSNOW", "CFRZR", "CICEP"):
        data = _fetch_grib_range(grib_url, var, "surface")
        if data is not None:
            results[var] = data

    if not results:
        logger.debug("No categorical vars found at %s", grib_url)
        return None

    return results


def _hrrr_url(dt: datetime) -> str:
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")
    return (
        f"{HRRR_BASE}/hrrr.{date_str}/conus/"
        f"hrrr.t{hour_str}z.wrfsfcf00.grib2"
    )


def _rap_url(dt: datetime) -> str:
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")
    return (
        f"{RAP_BASE}/rap.{date_str}/"
        f"rap.t{hour_str}z.awp130pgrbf00.grib2"
    )


def _candidate_run_times() -> list[datetime]:
    """Return recent hourly run times to try, most recent first."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [now - timedelta(hours=h) for h in range(1, 6)]


def fetch_hrrr_precip_type() -> Optional[Dict[str, bytes]]:
    """Fetch categorical precip records from the most recent available HRRR run."""
    for dt in _candidate_run_times():
        result = _try_nwp_run(HRRR_BASE, _hrrr_url, dt)
        if result:
            return result
    logger.warning("All HRRR attempts failed")
    return None


def fetch_rap_precip_type() -> Optional[Dict[str, bytes]]:
    """Fetch categorical precip records from the most recent available RAP run."""
    for dt in _candidate_run_times():
        result = _try_nwp_run(RAP_BASE, _rap_url, dt)
        if result:
            return result
    logger.warning("All RAP attempts failed")
    return None
