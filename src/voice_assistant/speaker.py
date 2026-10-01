"""Local speaker embedding and cosine verification."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sherpa_onnx
from .speaker_store import load_centroid, normalize

@dataclass(frozen=True)
class SpeakerVerificationResult:
    speaker_name: str
    authorized: bool
    centroid_score: float
    threshold: float

class SpeakerVerifier:
    def __init__(self, model_path, speaker_name, threshold, sample_rate=16000):
        self.model_path, self.speaker_name = Path(model_path), speaker_name
        self.threshold, self.sample_rate = threshold, sample_rate
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Speaker model not found: {self.model_path}")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.model_path), num_threads=2, provider="cpu")
        if not config.validate():
            raise RuntimeError("Invalid speaker embedding configuration")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, samples):
        stream = self._extractor.create_stream()
        stream.accept_waveform(self.sample_rate, np.ascontiguousarray(samples, dtype=np.float32))
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError("Not enough audio for speaker embedding")
        return normalize(np.asarray(self._extractor.compute(stream), dtype=np.float32))

    def verify(self, samples):
        score = float(np.dot(load_centroid(self.speaker_name), self.embed(samples)))
        return SpeakerVerificationResult(self.speaker_name, score >= self.threshold, score, self.threshold)
