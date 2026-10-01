from pathlib import Path

import numpy as np
import sherpa_onnx
class SileroVAD:
    """Stateful Silero VAD that accepts arbitrary-size float32 PCM chunks."""
    def __init__(self, model_path, sample_rate=16000, threshold=0.50, min_speech_seconds=0.25, min_silence_seconds=0.40, max_speech_seconds=30.0):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.min_silence_seconds = min_silence_seconds
        self.max_speech_seconds = max_speech_seconds
        self._pending = np.empty(0, dtype=np.float32)
        self._build()

    def _build(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {self.model_path}")
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.model_path)
        config.silero_vad.threshold = self.threshold
        config.silero_vad.min_speech_duration = self.min_speech_seconds
        config.silero_vad.min_silence_duration = self.min_silence_seconds
        config.silero_vad.max_speech_duration = self.max_speech_seconds
        config.sample_rate = self.sample_rate
        self.window_size = config.silero_vad.window_size
        self._detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def reset(self):
        self._pending = np.empty(0, dtype=np.float32)
        self._build()

    def is_speech(self, samples):
        audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if audio.size:
            self._pending = np.concatenate((self._pending, audio))
        while self._pending.size >= self.window_size:
            window = self._pending[:self.window_size]
            self._pending = self._pending[self.window_size:]
            self._detector.accept_waveform(window)
        while not self._detector.empty():
            self._detector.pop()
        return self._detector.is_speech_detected(), None
