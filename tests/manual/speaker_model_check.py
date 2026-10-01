from pathlib import Path

import sherpa_onnx


ROOT = Path(__file__).resolve().parents[2]
MODEL = str(ROOT / "models/speaker/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx")


config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=MODEL,
    num_threads=2,
    provider="cpu",
)


if not config.validate():
    raise RuntimeError("Speaker model config không hợp lệ")


extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)


print("Model loaded!")
print("Embedding dimension:", extractor.dim)