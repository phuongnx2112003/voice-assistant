import unittest
import numpy as np
from voice_assistant.aec import AcousticEchoCanceller


class AecTests(unittest.TestCase):
    def test_reduces_synthetic_delayed_echo(self):
        aec = AcousticEchoCanceller(sample_rate=16000, delay_ms=20, taps=128, mu=0.7)
        rng = np.random.default_rng(7)
        reference = rng.normal(0, 0.1, 16000).astype(np.float32)
        mic = np.concatenate((np.zeros(320, dtype=np.float32), 0.7 * reference))
        aec.feed_playback(reference, 16000)
        residual_blocks = []
        for offset in range(0, mic.size, 256):
            residual_blocks.append(aec.process_microphone(mic[offset:offset + 256]))
        residual = np.concatenate(residual_blocks)
        self.assertLess(np.mean(residual[-4000:] ** 2), np.mean(mic[-4000:] ** 2) * 0.5)

    def test_identifies_direct_echo_but_not_independent_audio(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(0, 0.1, 512).astype(np.float32)
        echo = AcousticEchoCanceller(sample_rate=16000, delay_ms=0)
        echo.feed_playback(reference, 16000)
        echo.process_microphone(0.5 * reference)
        self.assertTrue(echo.is_probable_echo())

        other = AcousticEchoCanceller(sample_rate=16000, delay_ms=0)
        other.feed_playback(reference, 16000)
        other.process_microphone(rng.normal(0, 0.1, 512).astype(np.float32))
        self.assertFalse(other.is_probable_echo())
