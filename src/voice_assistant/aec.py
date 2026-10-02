"""Lightweight local acoustic echo cancellation for full-duplex voice use."""

from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np


class AcousticEchoCanceller:
    """Block-NLMS echo canceller using the audio sent to the speaker as reference.

    It is deliberately local and dependency-free.  The delay is configurable
    because the physical speaker-to-microphone path varies by device.
    """

    def __init__(self, sample_rate: int = 16_000, delay_ms: float = 100.0, taps: int = 256, mu: float = 0.35):
        if sample_rate <= 0 or delay_ms < 0 or taps < 8 or not 0 < mu <= 1:
            raise ValueError("AEC configuration không hợp lệ")
        self.sample_rate = sample_rate
        self.taps = taps
        self.mu = mu
        self._lock = Lock()
        self._reference: deque[np.ndarray] = deque()
        self._available = 0
        self._history = np.zeros(taps - 1, dtype=np.float32)
        self._weights = np.zeros(taps, dtype=np.float32)
        self._reference_samples = 0
        self.last_echo_correlation = 0.0
        self.last_reference_rms = 0.0
        self.last_microphone_rms = 0.0
        self._append_reference(np.zeros(round(sample_rate * delay_ms / 1000), dtype=np.float32))

    @property
    def ready(self) -> bool:
        """True after enough playback reference has arrived for stable filtering."""
        return self._reference_samples >= self.sample_rate // 3

    def _append_reference(self, samples: np.ndarray) -> None:
        if samples.size:
            self._reference.append(samples)
            self._available += samples.size

    def feed_playback(self, samples: np.ndarray, sample_rate: int) -> None:
        """Provide a chunk that is about to be sent to the output device."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        if sample_rate != self.sample_rate:
            target_size = max(1, round(audio.size * self.sample_rate / sample_rate))
            positions = np.linspace(0, audio.size - 1, target_size, dtype=np.float32)
            audio = np.interp(positions, np.arange(audio.size), audio).astype(np.float32)
        with self._lock:
            self._append_reference(np.ascontiguousarray(audio))
            self._reference_samples += audio.size

    def _take_reference(self, count: int) -> np.ndarray:
        output = np.zeros(count, dtype=np.float32)
        offset = 0
        while offset < count and self._reference:
            head = self._reference[0]
            take = min(count - offset, head.size)
            output[offset:offset + take] = head[:take]
            offset += take
            if take == head.size:
                self._reference.popleft()
            else:
                self._reference[0] = head[take:]
            self._available -= take
        return output

    def process_microphone(self, samples: np.ndarray) -> np.ndarray:
        """Remove estimated speaker echo from one microphone block."""
        mic = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if not mic.size:
            return mic
        with self._lock:
            reference = self._take_reference(mic.size)
            self.last_microphone_rms = float(np.sqrt(np.mean(mic * mic)))
            self.last_reference_rms = float(np.sqrt(np.mean(reference * reference)))
            if not np.any(reference):
                self.last_echo_correlation = 0.0
                return mic.copy()
            correlation_denominator = float(np.linalg.norm(mic) * np.linalg.norm(reference))
            self.last_echo_correlation = (
                abs(float(np.dot(mic, reference))) / correlation_denominator
                if correlation_denominator > 1e-8 else 0.0
            )
            padded = np.concatenate((self._history, reference))
            windows = np.lib.stride_tricks.sliding_window_view(padded, self.taps)[:mic.size, ::-1]
            # Guard the adaptive update: room echo is nonlinear and occasional
            # microphone peaks must never make the filter diverge to NaN/Inf.
            if not np.isfinite(self._weights).all():
                self._weights.fill(0)
            estimate = windows @ self._weights
            residual = np.clip(mic - estimate, -1.0, 1.0)
            denominator = max(float(np.dot(reference, reference)), 1e-3)
            gradient = (windows.T @ residual) / denominator
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            gradient = np.clip(gradient, -0.10, 0.10)
            self._weights *= 0.999  # slow leakage avoids accumulating room-noise bias
            self._weights += self.mu * gradient.astype(np.float32)
            np.clip(self._weights, -1.5, 1.5, out=self._weights)
            self._history = padded[-(self.taps - 1):]
            return np.ascontiguousarray(residual, dtype=np.float32)

    def is_probable_echo(self, correlation_threshold: float = 0.35) -> bool:
        """Detect direct speaker leakage that remains after adaptive filtering."""
        return (
            self.last_reference_rms > 0.003
            and self.last_echo_correlation >= correlation_threshold
            and self.last_microphone_rms <= self.last_reference_rms * 1.5
        )
