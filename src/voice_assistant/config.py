from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
BLOCK_MS = 80
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
WAKE_THRESHOLD = 0.50
WAKE_MODEL = ROOT / 'models/wakeword/hey_kira.onnx'
VAD_MODEL = ROOT / 'models/vad/silero_vad.onnx'
STT_DIR = ROOT / 'models/stt/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09'
TTS_DIR = ROOT / 'models/tts'
SPEAKER_MODEL = ROOT / 'models/speaker/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx'
STT_THREADS = 2
INITIAL_SPEECH_TIMEOUT_SECONDS = 4.0
SILENCE_SECONDS = 1.0
MAX_UTTERANCE_SECONDS = 15.0
WAKE_COOLDOWN_SECONDS = 1.0
PARTIAL_DECODE_INTERVAL_SECONDS = 0.45

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.50"))
    wake_threshold: float = float(os.getenv("WAKE_THRESHOLD", str(WAKE_THRESHOLD)))
    speaker_threshold: float = float(os.getenv("SPEAKER_THRESHOLD", "0.60"))
    stt_threads: int = int(os.getenv("STT_THREADS", str(STT_THREADS)))
    initial_speech_timeout_seconds: float = float(os.getenv("INITIAL_SPEECH_TIMEOUT_SECONDS", str(INITIAL_SPEECH_TIMEOUT_SECONDS)))
    silence_seconds: float = float(os.getenv("SILENCE_SECONDS", str(SILENCE_SECONDS)))
    max_utterance_seconds: float = float(os.getenv("MAX_UTTERANCE_SECONDS", str(MAX_UTTERANCE_SECONDS)))

settings = Settings()
