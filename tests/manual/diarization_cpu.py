"""Run Sherpa-ONNX offline speaker diarization on a 16 kHz mono PCM WAV.

This manual diagnostic reports who spoke when. Labels are local to the input
recording and do not identify an enrolled user.

Example:
    ./scripts/download_sherpa_models.sh
    venv/bin/python tests/manual/diarization_cpu.py --audio samples/two_people.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import wave

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTATION_MODEL = ROOT / "models/diarization/sherpa-onnx-reverb-diarization-v1/model.onnx"
DEFAULT_EMBEDDING_MODEL = ROOT / "models/speaker/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
SAMPLE_RATE = 16_000


def read_pcm16_mono_wave(path: Path) -> np.ndarray:
    """Read the test's explicit 16 kHz, mono, 16-bit PCM input contract."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Audio must be mono. Convert it to mono before testing.")
        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Audio must be {SAMPLE_RATE} Hz; got {wav.getframerate()} Hz.")
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Audio must be an uncompressed 16-bit PCM WAV.")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def build_diarizer(args: argparse.Namespace) -> sherpa_onnx.OfflineSpeakerDiarization:
    pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        model=str(args.segmentation_model), window_shift_ratio=0.1
    )
    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=pyannote, num_threads=args.threads, provider="cpu"
    )
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(args.embedding_model), num_threads=args.threads, provider="cpu"
    )
    clustering = sherpa_onnx.FastClusteringConfig(
        num_clusters=args.clusters,
        threshold=args.threshold,
        compute_confidence=True,
    )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=clustering,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Sherpa diarization configuration is invalid")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="16 kHz mono PCM WAV")
    parser.add_argument("--segmentation-model", type=Path, default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per ONNX model")
    parser.add_argument(
        "--clusters", type=int, default=-1,
        help="Known speaker count; -1 detects it automatically.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Clustering threshold when --clusters=-1; tune on real recordings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, path in (
        ("Audio", args.audio),
        ("Segmentation model", args.segmentation_model),
        ("Embedding model", args.embedding_model),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.clusters == 0 or args.clusters < -1:
        raise ValueError("--clusters must be -1 or a positive integer")

    samples = np.ascontiguousarray(read_pcm16_mono_wave(args.audio), dtype=np.float32)
    duration = samples.size / SAMPLE_RATE
    diarizer = build_diarizer(args)
    print(f"Audio: {args.audio} ({duration:.2f}s)")
    print("Running offline CPU diarization...")
    started = time.perf_counter()
    result = diarizer.process(samples.tolist())
    elapsed = time.perf_counter() - started

    segments = result.sort_by_start_time()
    speaker_ids = sorted({segment.speaker for segment in segments})
    labels = ", ".join(f"speaker_{speaker_id}" for speaker_id in speaker_ids) or "none"
    print(f"Completed in {elapsed:.2f}s (RTF={elapsed / duration:.2f})")
    print(f"Detected speakers: {result.num_speakers} ({labels})")
    print("\n start    end      speaker      confidence")
    for segment in segments:
        confidence = getattr(segment, "confidence", None)
        confidence_text = f"{confidence:.3f}" if confidence is not None else "n/a"
        print(f" {segment.start:7.2f}  {segment.end:7.2f}  speaker_{segment.speaker:<3}  {confidence_text}")


if __name__ == "__main__":
    main()
