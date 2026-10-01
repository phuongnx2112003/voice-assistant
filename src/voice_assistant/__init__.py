"""Reusable local voice-processing components."""
from .service import VoiceResult, VoiceService
from .speaker import SpeakerVerificationResult, SpeakerVerifier
from .stt import VietnameseSTT
from .tts import VietnameseTTS
from .vad import SileroVAD

__all__ = [
    "SileroVAD",
    "SpeakerVerifier",
    "SpeakerVerificationResult",
    "VietnameseSTT",
    "VietnameseTTS",
    "VoiceResult",
    "VoiceService",
]
