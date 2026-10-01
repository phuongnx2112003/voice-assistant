#!/usr/bin/env bash
# Download the Sherpa-ONNX models used by this project into config.py locations.
# Safe to run repeatedly: complete models are skipped.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SPEAKER_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
DIARIZATION_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2"
TTS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-vi_VN-vais1000-medium.tar.bz2"

STT_DIR="$MODELS_DIR/stt/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
VAD_FILE="$MODELS_DIR/vad/silero_vad.onnx"
SPEAKER_FILE="$MODELS_DIR/speaker/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
DIARIZATION_DIR="$MODELS_DIR/diarization/sherpa-onnx-reverb-diarization-v1"
TTS_DIR="$MODELS_DIR/tts"

require_command() {
  command -v "$1" >/dev/null || {
    echo "Thiếu lệnh bắt buộc: $1" >&2
    exit 1
  }
}

fetch() {
  local url="$1"
  local output="$2"
  echo "↓ Downloading $(basename "$output")"
  curl --fail --location --retry 3 --retry-delay 2 --output "$output" "$url"
}

install_archive_dir() {
  local url="$1"
  local archive_name="$2"
  local extracted_name="$3"
  local destination="$4"
  local marker="$5"

  if [[ -f "$destination/$marker" ]]; then
    echo "✓ Already present: $destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Destination exists but is incomplete: $destination" >&2
    echo "Remove or repair it before downloading again." >&2
    exit 1
  fi

  local archive="$TEMP_DIR/$archive_name"
  fetch "$url" "$archive"
  tar -xjf "$archive" -C "$TEMP_DIR"
  [[ -d "$TEMP_DIR/$extracted_name" ]] || {
    echo "Archive did not contain expected directory: $extracted_name" >&2
    exit 1
  }
  mkdir -p "$(dirname "$destination")"
  mv "$TEMP_DIR/$extracted_name" "$destination"
  echo "✓ Installed: $destination"
}

install_single_file() {
  local url="$1"
  local destination="$2"
  if [[ -f "$destination" ]]; then
    echo "✓ Already present: $destination"
    return
  fi
  mkdir -p "$(dirname "$destination")"
  fetch "$url" "$destination"
  echo "✓ Installed: $destination"
}

require_command curl
require_command tar

install_archive_dir "$ASR_URL" \
  "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2" \
  "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09" \
  "$STT_DIR" "encoder.int8.onnx"
install_single_file "$VAD_URL" "$VAD_FILE"
install_single_file "$SPEAKER_URL" "$SPEAKER_FILE"
install_archive_dir "$DIARIZATION_URL" \
  "sherpa-onnx-reverb-diarization-v1.tar.bz2" \
  "sherpa-onnx-reverb-diarization-v1" \
  "$DIARIZATION_DIR" "model.onnx"
install_archive_dir "$TTS_URL" \
  "vits-piper-vi_VN-vais1000-medium.tar.bz2" \
  "vits-piper-vi_VN-vais1000-medium" \
  "$TTS_DIR" "vi_VN-vais1000-medium.onnx"

echo "All Sherpa-ONNX models are available under: $MODELS_DIR"
