#!/usr/bin/env bash
# setup.sh — install dependencies and start the MRMS viewer
set -e

echo "=== Installing Python dependencies ==="
pip3 install --break-system-packages --ignore-installed blinker \
  flask requests numpy scipy Pillow cfgrib eccodes

echo ""
echo "=== Starting MRMS Viewer on http://0.0.0.0:5000 ==="
echo "  Open http://localhost:5000 in your browser."
echo "  The first frame will be fetched automatically on startup."
echo ""
cd "$(dirname "$0")"
python3 app.py
