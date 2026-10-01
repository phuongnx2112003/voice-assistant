#!/usr/bin/env bash
# Install the optional Laya test dependencies without making the base voice
# assistant depend on OpenWakeWord's unavailable Python 3.12 tflite-runtime.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PYTHON_BIN" -m pip install "laya>=0.3.20"
