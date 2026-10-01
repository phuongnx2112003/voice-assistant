#!/usr/bin/env python3
"""CLI voice assistant: mic -> wake word -> speaker verification -> STT -> LLM -> TTS.

Required setup:
  1. Enroll the owner once:  python -m voice_assistant.cli.enroll_speaker
  2. Configure OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL in .env
  3. Start:                  python -m voice_assistant.cli.voice_cli

The process stops immediately when speaker verification fails, by design.
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from .. import config
from ..speaker_store import (
    SPEAKER_DATABASE,
    SPEAKER_MODEL,
    SPEAKER_NAME,
    SPEAKER_THRESHOLD,
    SPEAKER_VERIFY_SECONDS,
    normalize,
)
from ..endpoint import EndpointDetector
from ..stt import VietnameseSTT
from ..speaker import SpeakerVerifier
from ..tts import VietnameseTTS
from ..vad import SileroVAD
from ..service import VoiceService
from ..wakeword import WakeWord


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WAKE_MODEL = ROOT / "models/wakeword/hey_kira.onnx"
TTS_DIR = ROOT / "models/tts"
SILERO_VAD_MODEL = ROOT / "models/vad/silero_vad.onnx"


def load_enrolled_embeddings(speaker_name: str) -> list[np.ndarray]:
    """Load every normalized enrollment vector so all cosine scores are visible."""
    if not SPEAKER_DATABASE.is_file():
        raise LookupError(
            f"Chưa có database speaker: {SPEAKER_DATABASE}. Hãy chạy: python -m voice_assistant.cli.enroll_speaker trước."
        )

    with sqlite3.connect(SPEAKER_DATABASE) as connection:
        rows = connection.execute(
            """
            SELECT embedding_dim, embedding FROM speaker_embeddings
            WHERE speaker_name = ? ORDER BY id
            """,
            (speaker_name,),
        ).fetchall()

    if not rows:
        raise LookupError(
            f"Chưa có enrollment cho '{speaker_name}'. Hãy chạy: python -m voice_assistant.cli.enroll_speaker trước."
        )

    embeddings = []
    for dimension, blob in rows:
        embedding = np.frombuffer(blob, dtype=np.float32)
        if embedding.size != dimension:
            raise RuntimeError("Một embedding trong database bị hỏng")
        embeddings.append(normalize(embedding))
    return embeddings


def build_speaker_extractor() -> sherpa_onnx.SpeakerEmbeddingExtractor:
    if not SPEAKER_MODEL.is_file():
        raise FileNotFoundError(f"Không tìm thấy speaker model: {SPEAKER_MODEL}")
    extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(SPEAKER_MODEL), num_threads=2, provider="cpu"
    )
    if not extractor_config.validate():
        raise RuntimeError("Speaker embedding model config không hợp lệ")
    return sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)


def embedding_for(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor, samples: np.ndarray
) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(config.SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise RuntimeError("Không đủ audio để tạo speaker embedding")
    return normalize(np.asarray(extractor.compute(stream), dtype=np.float32))


def build_silero_vad() -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    """Create a fresh Silero VAD for one command recording."""
    if not SILERO_VAD_MODEL.is_file():
        raise FileNotFoundError(f"Không tìm thấy Silero VAD model: {SILERO_VAD_MODEL}")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = str(SILERO_VAD_MODEL)
    vad_config.silero_vad.threshold = 0.50
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.silero_vad.min_silence_duration = 0.40
    vad_config.silero_vad.max_speech_duration = config.MAX_UTTERANCE_SECONDS
    vad_config.sample_rate = config.SAMPLE_RATE
    window_size = vad_config.silero_vad.window_size
    return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30), window_size


def build_tts() -> tuple[sherpa_onnx.OfflineTts, sherpa_onnx.GenerationConfig]:
    required = ["vi_VN-vais1000-medium.onnx", "tokens.txt", "espeak-ng-data"]
    missing = [name for name in required if not (TTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"TTS model thiếu trong {TTS_DIR}: {missing}")
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(TTS_DIR / "vi_VN-vais1000-medium.onnx"),
                data_dir=str(TTS_DIR / "espeak-ng-data"),
                tokens=str(TTS_DIR / "tokens.txt"),
            ),
            num_threads=2,
            provider="cpu",
        )
    )
    if not tts_config.validate():
        raise RuntimeError("TTS config không hợp lệ")
    generation = sherpa_onnx.GenerationConfig()
    generation.sid = 0
    generation.speed = 1.0
    generation.silence_scale = 0.2
    return sherpa_onnx.OfflineTts(tts_config), generation


class VoiceCli:
    def __init__(self, args: argparse.Namespace):
        load_dotenv(ROOT / ".env")
        self.args = args
        self.wake = WakeWord(args.wake_model, args.wake_threshold)
        self.speaker = SpeakerVerifier(SPEAKER_MODEL, args.speaker_name, args.speaker_threshold)
        self.speaker_extractor = None
        self.enrollments = load_enrolled_embeddings(args.speaker_name)
        self.centroid = normalize(np.mean(self.enrollments, axis=0))
        self.service = VoiceService(
            SileroVAD(config.VAD_MODEL, sample_rate=config.SAMPLE_RATE, threshold=config.settings.vad_threshold),
            VietnameseSTT(config.STT_DIR, config.settings.stt_threads),
            self.speaker,
            VietnameseTTS(config.TTS_DIR),
        )
        self.llm = self._build_llm()

    @staticmethod
    def _build_llm() -> tuple[OpenAI, str]:
        missing = [key for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL") if not os.getenv(key)]
        if missing:
            raise RuntimeError(f"Thiếu biến trong .env: {', '.join(missing)}")
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"]), os.environ["OPENAI_MODEL"]

    def record_fixed_duration(self, seconds: float, label: str) -> np.ndarray:
        print(f"🎤 {label} trong {seconds:g} giây...", flush=True)
        audio = sd.rec(
            int(seconds * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.args.input_device,
        )
        sd.wait()
        samples = audio[:, 0]
        if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0.0) < 1e-4:
            raise RuntimeError("Không nhận được tín hiệu microphone")
        return samples

    def wait_for_wake_word(self) -> None:
        print(f"🎧 Đang chờ wake word: {Path(self.args.wake_model).stem} (Ctrl+C để thoát)")
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=config.BLOCK_SAMPLES,
            device=self.args.input_device,
        ) as stream:
            while True:
                samples, _ = stream.read(config.BLOCK_SAMPLES)
                score, _ = self.wake.process(samples[:, 0])
                sys.stdout.write(f"\rwake score={score:.3f}  ")
                sys.stdout.flush()
                if score >= self.args.wake_threshold:
                    print(f"\n🔥 Wake word detected (score={score:.4f})")
                    self.wake.reset()
                    return

    def verify_speaker(self) -> bool:
        samples = self.record_fixed_duration(self.args.verify_seconds, "Bắt đầu nói tự nhiên")
        candidate = self.speaker.embed(samples)
        enrollment_scores = [float(np.dot(vector, candidate)) for vector in self.enrollments]
        centroid_score = float(np.dot(self.centroid, candidate))
        print("\n🔐 Speaker verification")
        for index, score in enumerate(enrollment_scores, start=1):
            print(f"  vector #{index}: cosine score = {score:.4f}")
        print(f"  centroid score : {centroid_score:.4f}")
        print(f"  threshold      : {self.args.speaker_threshold:.4f}")
        authorized = centroid_score >= self.args.speaker_threshold
        print("✅ Đúng người dùng." if authorized else "❌ Không khớp người dùng. Dừng chương trình.")
        return authorized

    def record_command(self) -> np.ndarray:
        print("\n🎤 Hãy nói yêu cầu của bạn...", flush=True)
        chunks: list[np.ndarray] = []
        vad = SileroVAD(config.VAD_MODEL, sample_rate=config.SAMPLE_RATE, threshold=config.settings.vad_threshold, max_speech_seconds=config.settings.max_utterance_seconds)
        vad_window_size = vad.window_size
        endpoint = EndpointDetector(
            config.settings.initial_speech_timeout_seconds,
            config.settings.silence_seconds,
        )
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=vad_window_size,
            device=self.args.input_device,
        ) as stream:
            while endpoint.elapsed < config.settings.max_utterance_seconds:
                samples, _ = stream.read(config.BLOCK_SAMPLES)
                chunk = samples[:, 0].copy()
                chunks.append(chunk)
                speech, _ = vad.is_speech(chunk)
                endpoint.observe(len(chunk) / config.SAMPLE_RATE, speech)
                if endpoint.should_stop():
                    break
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    def ask_llm(self, text: str) -> str:
        client, model = self.llm
        response = client.responses.create(
            model=model,
            instructions=(
                "Bạn là Kira, trợ lý giọng nói thân thiện. Luôn trả lời bằng tiếng Việt "
                "có dấu, tự nhiên, ngắn gọn và phù hợp để đọc thành tiếng."
            ),
            input=text,
        )
        return response.output_text.strip()

    def speak(self, text: str) -> None:
        audio = self.service.synthesize(text)
        if len(audio.samples) == 0:
            raise RuntimeError("TTS không tạo được audio")
        sd.play(audio.samples, samplerate=audio.sample_rate, device=self.args.output_device, blocking=True)

    def run(self) -> None:
        self.wait_for_wake_word()
        if not self.verify_speaker():
            return
        command_audio = self.record_command()
        result = self.service.process(command_audio, config.SAMPLE_RATE)
        transcript = result.transcript
        if not result.speech_detected or not transcript:
            print("⚠️ STT không nhận dạng được yêu cầu; dừng chương trình.")
            return
        print(f"\n📝 STT: {transcript}")
        print("🤖 Đang hỏi LLM...", flush=True)
        answer = self.ask_llm(transcript)
        if not answer:
            print("⚠️ LLM không trả về nội dung.")
            return
        print(f"🤖 LLM: {answer}")
        print("🔊 TTS đang phát...", flush=True)
        self.speak(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mic -> wake word -> speaker verification -> STT -> LLM -> TTS")
    parser.add_argument("--input-device", type=int, default=None, help="ID microphone (mặc định: microphone hệ thống)")
    parser.add_argument("--output-device", type=int, default=None, help="ID loa (mặc định: loa hệ thống)")
    parser.add_argument("--wake-model", type=Path, default=DEFAULT_WAKE_MODEL)
    parser.add_argument("--wake-threshold", type=float, default=config.settings.wake_threshold)
    parser.add_argument("--speaker-name", default=SPEAKER_NAME)
    parser.add_argument("--speaker-threshold", type=float, default=config.settings.speaker_threshold)
    parser.add_argument("--verify-seconds", type=float, default=SPEAKER_VERIFY_SECONDS)
    return parser.parse_args()


def main() -> None:
    try:
        VoiceCli(parse_args()).run()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    except Exception as error:
        print(f"\n❌ Lỗi: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
