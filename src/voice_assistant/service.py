"""Framework-independent orchestration for a voice/audio request."""
from dataclasses import dataclass
from threading import RLock
import numpy as np
from .speaker import SpeakerVerifier, SpeakerVerificationResult
from .stt import VietnameseSTT
from .tts import VietnameseTTS
from .vad import SileroVAD

@dataclass(frozen=True)
class VoiceResult:
    speech_detected: bool
    transcript: str
    speaker: SpeakerVerificationResult | None

class VoiceService:
    """Use from HTTP/WebSocket workers; callers supply float32, 16 kHz PCM."""
    def __init__(self, vad: SileroVAD, stt: VietnameseSTT, speaker: SpeakerVerifier | None = None, tts: VietnameseTTS | None = None):
        self.vad, self.stt, self.speaker, self.tts = vad, stt, speaker, tts
        self._lock = RLock()

    def process(self, samples, sample_rate=16000, verify_speaker=False):
        """Process one complete request. Calls are serialized because model sessions are stateful."""
        with self._lock:
            return self._process(samples, sample_rate, verify_speaker)

    def _process(self, samples, sample_rate=16000, verify_speaker=False):
        audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if sample_rate != self.vad.sample_rate:
            raise ValueError(f"Expected {self.vad.sample_rate} Hz audio, got {sample_rate} Hz")
        self.vad.reset()
        speech_detected = False
        for offset in range(0, audio.size, self.vad.window_size):
            active, _ = self.vad.is_speech(audio[offset:offset + self.vad.window_size])
            speech_detected = speech_detected or active
        if not speech_detected:
            return VoiceResult(False, "", None)
        speaker = self.speaker.verify(audio) if verify_speaker and self.speaker else None
        if speaker is not None and not speaker.authorized:
            return VoiceResult(True, "", speaker)
        return VoiceResult(True, self.stt.transcribe(audio, sample_rate), speaker)

    def synthesize(self, text):
        if self.tts is None:
            raise RuntimeError("TTS is not configured")
        return self.tts.synthesize(text)
