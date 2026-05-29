#!/usr/bin/env python3
"""
app.py - MRMS Precipitation Viewer Flask server.

Serves a Leaflet-based interactive map with MRMS PrecipRate overlaid,
coloured by precip type sourced from HRRR → RAP → MRMS fallback.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file, abort

import fetch
import process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

FRAMES_DIR = Path(__file__).parent / "frames"
FRAMES_DIR.mkdir(exist_ok=True)

# ── Shared state (protected by _lock) ──────────────────────────────────────
_lock = threading.Lock()
_state: dict = {
    "frame_path": None,
    "metadata": None,
    "processing": False,
    "error": None,
    "updated_at": None,
    "ptype_source": "none",
}


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

def _build_frame() -> None:
    """Fetch latest data, generate frame PNG, update _state.  Thread-safe."""
    with _lock:
        if _state["processing"]:
            logger.info("Frame generation already in progress; skipping.")
            return
        _state["processing"] = True
        _state["error"] = None

    prev_path = None
    with _lock:
        prev_path = _state["frame_path"]

    try:
        # 1. Latest MRMS PrecipRate
        rate_url = fetch.find_latest_mrms_file("PrecipRate")
        if rate_url is None:
            raise RuntimeError("Could not locate latest MRMS PrecipRate file on S3")
        rate_bytes = fetch.download_grib2(rate_url)
        if rate_bytes is None:
            raise RuntimeError("Failed to download MRMS PrecipRate")

        # 2. Precip type: HRRR → RAP → MRMS fallback
        #    Always pre-fetch MRMS PrecipFlag so generate_frame can fall back
        #    to it even when NWP data is available but all-zero (f=00 spin-up).
        ptype_cat = None
        ptype_source = "none"

        logger.info("Pre-fetching MRMS PrecipFlag as fallback ...")
        mrms_ptype_bytes = None
        for product in ("PrecipType", "PrecipFlag"):
            pt_url = fetch.find_latest_mrms_file(product)
            if pt_url:
                mrms_ptype_bytes = fetch.download_grib2(pt_url)
                if mrms_ptype_bytes:
                    break

        logger.info("Trying HRRR for precip type ...")
        ptype_cat = fetch.fetch_hrrr_precip_type()
        if ptype_cat:
            ptype_source = "HRRR"
        else:
            logger.info("HRRR failed; trying RAP ...")
            ptype_cat = fetch.fetch_rap_precip_type()
            if ptype_cat:
                ptype_source = "RAP"

        # 3. Generate frame
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = FRAMES_DIR / f"frame_{ts}.png"

        metadata = process.generate_frame(
            rate_bytes=rate_bytes,
            ptype_cat=ptype_cat,
            mrms_ptype_bytes=mrms_ptype_bytes,
            output_path=str(out_path),
        )
        metadata["timestamp"] = ts
        # Use the source that was actually applied inside generate_frame
        ptype_source = metadata.pop("ptype_source_used", ptype_source)
        metadata["ptype_source"] = ptype_source
        metadata["rate_url"] = rate_url

        # 4. Update state and delete previous frame
        with _lock:
            _state["frame_path"] = str(out_path)
            _state["metadata"] = metadata
            _state["updated_at"] = ts
            _state["ptype_source"] = ptype_source
            _state["error"] = None

        if prev_path and prev_path != str(out_path):
            try:
                os.unlink(prev_path)
                logger.info("Deleted previous frame: %s", prev_path)
            except OSError:
                pass

        logger.info("Frame ready: %s  ptype_source=%s", out_path.name, ptype_source)

    except Exception as exc:
        logger.exception("Frame generation failed: %s", exc)
        with _lock:
            _state["error"] = str(exc)
    finally:
        with _lock:
            _state["processing"] = False


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/frame-info")
def frame_info():
    with _lock:
        info = {
            "frame_ready": _state["frame_path"] is not None,
            "processing": _state["processing"],
            "error": _state["error"],
            "updated_at": _state["updated_at"],
            "ptype_source": _state["ptype_source"],
            "metadata": _state["metadata"],
        }
    return jsonify(info)


@app.route("/api/frame.png")
def serve_frame():
    with _lock:
        path = _state["frame_path"]

    if path is None or not os.path.exists(path):
        abort(404)

    # no-cache so the browser always re-fetches after a refresh
    response = send_file(path, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/refresh", methods=["POST"])
def refresh():
    t = threading.Thread(target=_build_frame, daemon=True)
    t.start()
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# Startup: generate initial frame in background
# ---------------------------------------------------------------------------

def _startup_frame():
    time.sleep(1)   # let Flask bind first
    _build_frame()


if __name__ == "__main__":
    t = threading.Thread(target=_startup_frame, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
