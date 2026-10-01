"""Verify a voice sample against local enrollment in the project SQLite DB."""

import numpy as np
import sherpa_onnx
import sounddevice as sd

from ..speaker_store import (
    SAMPLE_RATE,
    SPEAKER_DATABASE,
    SPEAKER_MODEL,
    SPEAKER_NAME,
    SPEAKER_THRESHOLD,
    SPEAKER_VERIFY_SECONDS,
    load_centroid,
    normalize,
)


def build_extractor() -> sherpa_onnx.SpeakerEmbeddingExtractor:
    if not SPEAKER_MODEL.is_file():
        raise FileNotFoundError(f"Không tìm thấy speaker model: {SPEAKER_MODEL}")
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(SPEAKER_MODEL), num_threads=2, provider="cpu"
    )
    if not config.validate():
        raise RuntimeError("Speaker model config không hợp lệ")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def compute_embedding(extractor: sherpa_onnx.SpeakerEmbeddingExtractor, samples: np.ndarray) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise RuntimeError("Không đủ audio để tạo speaker embedding")
    return normalize(np.asarray(extractor.compute(stream), dtype=np.float32))


def record(seconds: float) -> np.ndarray:
    print(f"🎤 Đang thu {seconds:g}s...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    samples = audio[:, 0]
    if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0.0) < 1e-4:
        raise RuntimeError("Không nhận được tín hiệu microphone. Hãy kiểm tra microphone rồi thử lại.")
    return samples


def main():
    owner_embedding = load_centroid(SPEAKER_NAME)
    extractor = build_extractor()
    print(f"=== SPEAKER VERIFICATION: '{SPEAKER_NAME}' ===")
    print(f"Database: {SPEAKER_DATABASE}")
    print(f"Ngưỡng cosine similarity: {SPEAKER_THRESHOLD:.2f}")

    while True:
        try:
            input("\nNhấn ENTER rồi nói (Ctrl+C để thoát)...")
            embedding = compute_embedding(extractor, record(SPEAKER_VERIFY_SECONDS))
            score = float(np.dot(owner_embedding, embedding))
            print(f"Similarity: {score:.4f}")
            print("✅ OWNER" if score >= SPEAKER_THRESHOLD else "❌ NOT OWNER")
        except KeyboardInterrupt:
            print("\nĐã dừng speaker verification.")
            return
        except Exception as error:
            print(f"⚠️ Không thể kiểm tra mẫu này: {error}")


if __name__ == "__main__":
    main()
