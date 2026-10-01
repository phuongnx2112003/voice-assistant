import unittest
import numpy as np
from voice_assistant.service import VoiceService
from voice_assistant.speaker import SpeakerVerificationResult

class FakeVAD:
    sample_rate = 16000
    window_size = 4
    def reset(self): pass
    def is_speech(self, samples): return bool(np.any(samples)), None

class FakeSTT:
    def transcribe(self, samples, sample_rate): return "xin chào"

class FakeSpeaker:
    def __init__(self, authorized=True): self.authorized = authorized
    def verify(self, samples): return SpeakerVerificationResult("owner", self.authorized, 0.9 if self.authorized else 0.1, 0.6)

class VoiceServiceTests(unittest.TestCase):
    def test_silence_skips_stt(self):
        result = VoiceService(FakeVAD(), FakeSTT()).process(np.zeros(16, dtype=np.float32))
        self.assertFalse(result.speech_detected)
        self.assertEqual(result.transcript, "")

    def test_authorized_voice_is_transcribed(self):
        result = VoiceService(FakeVAD(), FakeSTT(), FakeSpeaker()).process(np.ones(16, dtype=np.float32), verify_speaker=True)
        self.assertTrue(result.speaker.authorized)
        self.assertEqual(result.transcript, "xin chào")

    def test_unauthorized_voice_is_not_transcribed(self):
        result = VoiceService(FakeVAD(), FakeSTT(), FakeSpeaker(False)).process(np.ones(16, dtype=np.float32), verify_speaker=True)
        self.assertFalse(result.speaker.authorized)
        self.assertEqual(result.transcript, "")
