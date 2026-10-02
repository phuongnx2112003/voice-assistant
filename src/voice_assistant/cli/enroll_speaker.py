"""Enroll local owner voice embeddings into the project SQLite database."""

import numpy as np
import sherpa_onnx
import sounddevice as sd

from ..speaker_store import (
    SAMPLE_RATE,
    SPEAKER_DATABASE,
    SPEAKER_ENROLL_SAMPLES,
    SPEAKER_ENROLL_SECONDS,
    SPEAKER_ENROLL_MIN_SIMILARITY,
    SPEAKER_MIN_SAMPLE_SECONDS,
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
    print("Nói liên tục một câu tự nhiên, tránh im lặng và tiếng ồn lớn.")
    print(
        f"Mỗi mẫu cần dài ít nhất {SPEAKER_MIN_SAMPLE_SECONDS:g}s; "
        f"mẫu lệch dưới similarity {SPEAKER_ENROLL_MIN_SIMILARITY:.2f} sẽ phải thu lại."
    )

    embeddings = []
    while len(embeddings) < SPEAKER_ENROLL_SAMPLES:
        index = len(embeddings) + 1
        input(f"Nhấn ENTER để thu mẫu {index}/{SPEAKER_ENROLL_SAMPLES}...")
        samples = record(SPEAKER_ENROLL_SECONDS)
        if samples.size / SAMPLE_RATE < SPEAKER_MIN_SAMPLE_SECONDS:
            print("⚠️ Mẫu quá ngắn; hãy nói lâu hơn rồi thử lại.")
            continue
        embedding = compute_embedding(extractor, samples)
        if embeddings:
            centroid = normalize(np.mean(embeddings, axis=0))
            score = float(np.dot(centroid, embedding))
            print(f"  similarity với các mẫu đã nhận: {score:.4f}")
            if score < SPEAKER_ENROLL_MIN_SIMILARITY:
                print("⚠️ Mẫu lệch nhiều; hãy thu lại ở nơi yên hơn và nói liên tục.")
                continue
        embeddings.append(embedding)
        print(f"✓ Đã nhận mẫu {index}; embedding dimension: {embedding.size}")

    count = replace_embeddings(SPEAKER_NAME, embeddings)
    print(f"\n✅ Đã lưu {count} embedding vào database local: {SPEAKER_DATABASE}")


if __name__ == "__main__":
    main()
