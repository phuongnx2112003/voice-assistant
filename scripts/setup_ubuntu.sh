#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev portaudio19-dev libsndfile1 wget bzip2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -e .
# Optional developer tools: pip install -r requirements-dev.txt
./scripts/download_sherpa_models.sh
echo 'Setup complete. Add models/wakeword/hey_phuong.onnx, then: source .venv/bin/activate && python -m voice_assistant'
