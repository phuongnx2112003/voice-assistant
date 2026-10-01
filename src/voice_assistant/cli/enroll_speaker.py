"""Enroll local owner voice embeddings into the project SQLite database."""

import numpy as np
import sherpa_onnx
import sounddevice as sd

from ..speaker_store import (
    SAMPLE_RATE,
    SPEAKER_DATABASE,
    SPEAKER_ENROLL_SAMPLES,
    SPEAKER_ENROLL_SECONDS,
    SPEAKER_MODEL,
    SPEAKER_NAME,
    normalize,
    replace_embeddings,
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
    print(f"🎤 Nói tự nhiên trong khoảng {seconds:g} giây...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    samples = audio[:, 0]
    if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0.0) < 1e-4:
        raise RuntimeError("Không nhận được tín hiệu microphone. Hãy kiểm tra microphone rồi thử lại.")
    return samples


def main():
    extractor = build_extractor()
    print(f"=== ENROLL '{SPEAKER_NAME}' ===")
    print("Nên thu ở các tư thế/khoảng cách tự nhiên khác nhau, nhưng tránh im lặng và tiếng ồn lớn.")

    embeddings = []
    for index in range(SPEAKER_ENROLL_SAMPLES):
        input(f"Nhấn ENTER để thu mẫu {index + 1}/{SPEAKER_ENROLL_SAMPLES}...")
        embedding = compute_embedding(extractor, record(SPEAKER_ENROLL_SECONDS))
        embeddings.append(embedding)
        print(f"✓ Đã nhận mẫu {index + 1}; embedding dimension: {embedding.size}")

    count = replace_embeddings(SPEAKER_NAME, embeddings)
    print(f"\n✅ Đã lưu {count} embedding vào database local: {SPEAKER_DATABASE}")


if __name__ == "__main__":
    main()
