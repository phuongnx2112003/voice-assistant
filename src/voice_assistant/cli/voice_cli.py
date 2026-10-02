#!/usr/bin/env python3
"""CLI voice assistant with wake-word or continuous owner-only activation.

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
import queue
import threading
import time
import re
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from .. import config
from ..aec import AcousticEchoCanceller
from ..speaker_store import (
    SPEAKER_DATABASE,
    SPEAKER_MODEL,
    SPEAKER_MIN_SAMPLE_SECONDS,
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


def text_for_speech(text: str) -> str:
    """Keep LLM Markdown for display, but turn it into natural text for TTS."""
    # Preserve the label while removing a Markdown URL: [OpenAI](url) -> OpenAI.
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"[-*_]{3,}", line):
            continue
        # Keep the content of headings, quotes, and lists, but drop their markers.
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        # Bold, italic, inline code, strikethrough, and escaped Markdown symbols.
        line = re.sub(r"[*_`~]", "", line)
        line = line.replace("\\", "")
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines).strip()


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
        self.wake = (
            WakeWord(args.wake_model, args.wake_threshold)
            if args.activation_mode == "wake"
            else None
        )
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
        # Set only while sounddevice is actively playing synthesized speech.
        # The continuous VAD loop uses this to implement barge-in.
        self._tts_playing = threading.Event()
        self._tts_cancel = threading.Event()
        self._tts_stream_lock = threading.Lock()
        self._tts_output_stream: sd.OutputStream | None = None
        self.aec = AcousticEchoCanceller(delay_ms=args.aec_delay_ms) if args.aec else None
        self._last_audio_debug_at = 0.0
        self._last_audio_metrics = ""
        self._tts_input_suppressed_notice = False
        self._last_input_rms = 0.0

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

    def _synthesize_for_speech(self, text: str):
        speech_text = text_for_speech(text)
        if not speech_text:
            raise RuntimeError("Phản hồi chỉ chứa Markdown, không có nội dung để đọc")
        audio = self.service.synthesize(speech_text)
        if len(audio.samples) == 0:
            raise RuntimeError("TTS không tạo được audio")
        return audio

    def _start_tts_playback(self, audio) -> threading.Thread:
        """Use a dedicated output stream so microphone capture remains live."""
        self._tts_cancel.clear()

        def playback() -> None:
            self._tts_playing.set()
            self._tts_input_suppressed_notice = False
            stream = None
            try:
                samples = np.ascontiguousarray(audio.samples, dtype=np.float32).reshape(-1, 1)
                with sd.OutputStream(
                    samplerate=audio.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.args.output_device,
                ) as stream:
                    with self._tts_stream_lock:
                        self._tts_output_stream = stream
                    for offset in range(0, samples.shape[0], 1024):
                        if self._tts_cancel.is_set():
                            return
                        chunk = samples[offset:offset + 1024]
                        if self.aec is not None:
                            self.aec.feed_playback(chunk[:, 0], audio.sample_rate)
                        stream.write(chunk)
            except Exception as error:
                if not self._tts_cancel.is_set():
                    print(f"\n⚠️ TTS playback lỗi: {error}", flush=True)
            finally:
                with self._tts_stream_lock:
                    if self._tts_output_stream is stream:
                        self._tts_output_stream = None
                self._tts_playing.clear()

        thread = threading.Thread(target=playback, name="tts-playback", daemon=True)
        thread.start()
        return thread

    def _stop_tts_playback(self) -> None:
        """Request a clean stop at the next small output block.

        Calling PortAudio abort() from the VAD thread caused ALSA XRUN errors on
        some Linux devices.  The writer checks this event every 1024 samples
        (about 64 ms at 16 kHz), then closes the stream normally.
        """
        self._tts_cancel.set()

    def _log_audio_debug(self, raw: np.ndarray, processed: np.ndarray) -> None:
        """Print signal levels needed to diagnose false barge-in without recording audio."""
        raw_rms = float(np.sqrt(np.mean(np.square(raw, dtype=np.float64)))) if raw.size else 0.0
        self._last_input_rms = raw_rms
        processed_rms = float(np.sqrt(np.mean(np.square(processed, dtype=np.float64)))) if processed.size else 0.0
        if self.aec is None:
            self._last_audio_metrics = f"aec=off mic_rms={raw_rms:.5f}"
        else:
            self._last_audio_metrics = (
                f"aec=on ready={self.aec.ready} mic_rms={raw_rms:.5f} "
                f"residual_rms={processed_rms:.5f} ref_rms={self.aec.last_reference_rms:.5f} "
                f"corr={self.aec.last_echo_correlation:.3f} "
                f"probable_echo={self.aec.is_probable_echo(self.args.aec_echo_correlation)}"
            )
        # RMS is needed by Barge-in regardless of debug mode.  Only printing is
        # conditional; returning before the measurement made normal Barge-in
        # impossible when --audio-debug was omitted.
        if not self.args.audio_debug or not self._tts_playing.is_set():
            return
        now = time.monotonic()
        if now - self._last_audio_debug_at < self.args.audio_debug_interval:
            return
        self._last_audio_debug_at = now
        if self.aec is None:
            print(
                f"\n🔎 audio-debug tts=on {self._last_audio_metrics}",
                flush=True,
            )
            return
        print(
            f"\n🔎 audio-debug tts=on {self._last_audio_metrics}",
            flush=True,
        )

    def speak(self, text: str) -> None:
        self._start_tts_playback(self._synthesize_for_speech(text)).join()

    def interrupt_tts_for_barge_in(self) -> bool:
        """Immediately stop playback when a VAD hears new speech."""
        if not self.args.barge_in or not self._tts_playing.is_set():
            return False
        # Clear before stopping output: the microphone can still contain a few
        # callback blocks of the interrupted TTS, which must not trigger a
        # second barge-in while sd.play() is unwinding.
        self._tts_playing.clear()
        try:
            self._stop_tts_playback()
        except Exception as error:
            print(f"\n⚠️ Không thể ngắt TTS: {error}", flush=True)
            return False
        print("\n🛑 Barge-in: đã ngắt TTS để nghe câu mới.", flush=True)
        return True

    def speak_with_wake_barge_in(self, text: str) -> np.ndarray | None:
        """Play a response while listening for one VAD-finalized interruption.

        A returned segment becomes the next turn in the already authorized wake
        session, so the user does not have to say the wake word again.
        """
        audio = self._synthesize_for_speech(text)

        vad, window_size = build_silero_vad()
        playback_thread = self._start_tts_playback(audio)
        barge_in_started = False
        loud_barge_in_seconds = 0.0
        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=window_size,
                device=self.args.input_device,
            ) as stream:
                while playback_thread.is_alive():
                    block, _ = stream.read(window_size)
                    raw_microphone = block[:, 0].copy()
                    microphone = raw_microphone
                    if self.aec is not None:
                        microphone = self.aec.process_microphone(microphone)
                        if self._tts_playing.is_set() and (
                            not self.aec.ready
                            or self.aec.is_probable_echo(self.args.aec_echo_correlation)
                        ):
                            continue
                    self._log_audio_debug(raw_microphone, microphone)
                    vad.accept_waveform(microphone)
                    speech = vad.is_speech_detected()
                    microphone_rms = self._last_input_rms
                    if speech and microphone_rms >= self.args.barge_in_rms:
                        loud_barge_in_seconds += window_size / config.SAMPLE_RATE
                    elif not speech:
                        loud_barge_in_seconds = 0.0
                    if (
                        speech
                        and not barge_in_started
                        and loud_barge_in_seconds >= self.args.barge_in_min_active_seconds
                    ):
                        # Do not stop on a single VAD hit. Keep listening until
                        # the user is clearly speaking, then finalize below.
                        self.interrupt_tts_for_wake_barge_in()
                        barge_in_started = True
                    while not vad.empty():
                        segment = vad.front
                        samples = np.ascontiguousarray(segment.samples, dtype=np.float32).copy()
                        vad.pop()
                        if barge_in_started:
                            playback_thread.join(timeout=1)
                            return samples
        finally:
            if playback_thread.is_alive():
                self._stop_tts_playback()
            playback_thread.join(timeout=1)
        return None

    def interrupt_tts_for_wake_barge_in(self) -> None:
        self._tts_playing.clear()
        try:
            self._stop_tts_playback()
        except Exception as error:
            print(f"\n⚠️ Không thể ngắt TTS: {error}", flush=True)
            return
        print("\n🛑 Barge-in: đã ngắt TTS để nghe câu mới.", flush=True)

    def answer_for_audio(self, command_audio: np.ndarray) -> str | None:
        result = self.service.process(command_audio, config.SAMPLE_RATE)
        transcript = result.transcript
        if not result.speech_detected or not transcript:
            print("⚠️ STT không nhận dạng được yêu cầu; dừng chương trình.")
            return None
        print(f"\n📝 STT: {transcript}")
        print("🤖 Đang hỏi LLM...", flush=True)
        answer = self.ask_llm(transcript)
        if not answer:
            print("⚠️ LLM không trả về nội dung.")
            return None
        print(f"🤖 LLM: {answer}")
        return answer

    def run(self) -> None:
        if self.args.activation_mode == "continuous":
            self.run_continuous()
            return
        self.wait_for_wake_word()
        if not self.verify_speaker():
            return
        command_audio = self.record_command()
        while command_audio.size:
            answer = self.answer_for_audio(command_audio)
            if not answer:
                return
            print("🔊 TTS đang phát...", flush=True)
            if not self.args.barge_in:
                self.speak(answer)
                return
            command_audio = self.speak_with_wake_barge_in(answer)
            if command_audio is None:
                return

    def run_continuous(self) -> None:
        """Use microphone callback -> VAD -> worker without a wake-word model."""
        vad, window_size = build_silero_vad()
        # Keep each block's level with its audio. The callback and VAD loop run
        # on different threads, so reading a mutable "latest RMS" value in the
        # VAD loop can refer to a later, unrelated microphone block.
        audio_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(
            maxsize=max(1, int(self.args.audio_queue_seconds * config.SAMPLE_RATE / window_size))
        )
        segment_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.args.segment_queue_size)
        last_overflow_notice = 0.0

        def microphone_callback(indata, frames, time_info, status) -> None:
            nonlocal last_overflow_notice
            del frames, time_info
            if status:
                print(f"\n⚠️ Microphone: {status}", flush=True)
            raw_block = indata[:, 0].copy()
            block = raw_block
            # In half-duplex mode, TTS is not a user turn.  Previously
            # --no-barge-in merely avoided stopping playback but still fed
            # microphone blocks to VAD, letting headphone leakage/noise become
            # a queued request as soon as TTS finished.
            if self._tts_playing.is_set() and not self.args.barge_in:
                self._log_audio_debug(raw_block, raw_block)
                if not self._tts_input_suppressed_notice:
                    print("\n🔇 Bỏ audio microphone trong lúc TTS phát (barge-in tắt).", flush=True)
                    self._tts_input_suppressed_notice = True
                return
            if self.aec is not None:
                block = self.aec.process_microphone(block)
            self._log_audio_debug(raw_block, block)
            if self.aec is not None:
                if self._tts_playing.is_set() and (
                    not self.aec.ready
                    or self.aec.is_probable_echo(self.args.aec_echo_correlation)
                ):
                    return
            block_rms = self._last_input_rms
            try:
                audio_queue.put_nowait((block, block_rms))
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                    audio_queue.put_nowait((block, block_rms))
                    now = time.monotonic()
                    if now - last_overflow_notice >= 1.0:
                        print("\n⚠️ Audio queue đầy; bỏ block cũ.", flush=True)
                        last_overflow_notice = now
                except queue.Empty:
                    pass

        def process_segment(samples: np.ndarray) -> None:
            duration = samples.size / config.SAMPLE_RATE
            print(f"\n🎙️ VAD chốt đoạn: {duration:.2f}s", flush=True)
            if self.args.continuous_verify_speaker and duration < SPEAKER_MIN_SAMPLE_SECONDS:
                print(
                    f"⚠️ Bỏ qua đoạn ngắn hơn {SPEAKER_MIN_SAMPLE_SECONDS:g}s; "
                    "chưa đủ ổn định để xác minh giọng.",
                    flush=True,
                )
                return
            transcript = self.service.stt.transcribe(samples, config.SAMPLE_RATE)
            if not transcript:
                print("⚠️ STT không tạo được text.", flush=True)
                return
            print(f"📝 STT: {transcript}", flush=True)
            if self.args.continuous_verify_speaker:
                try:
                    verification = self.speaker.verify(samples)
                except ValueError as error:
                    print(f"⚠️ Speaker embedding chưa đủ tin cậy: {error}", flush=True)
                    return
                print(f"🔐 owner score={verification.centroid_score:.3f} (threshold={verification.threshold:.2f})", flush=True)
                if not verification.authorized:
                    print("⛔ Non-owner: không gửi STT sang LLM/TTS.", flush=True)
                    return
            else:
                print("🔓 Speaker verification: OFF", flush=True)
            print("🤖 Đang hỏi LLM...", flush=True)
            answer = self.ask_llm(transcript)
            if not answer:
                print("⚠️ LLM không trả về nội dung.", flush=True)
                return
            print(f"🤖 LLM: {answer}", flush=True)
            print("🔊 TTS đang phát...", flush=True)
            self.speak(answer)

        def worker() -> None:
            while True:
                samples = segment_queue.get()
                try:
                    if samples is None:
                        return
                    process_segment(samples)
                except Exception as error:
                    print(f"⚠️ Không xử lý được segment: {error}", flush=True)
                finally:
                    segment_queue.task_done()

        threading.Thread(target=worker, name="continuous-owner-worker", daemon=True).start()
        print("=== CONTINUOUS OWNER MODE ===")
        verification_status = "ON" if self.args.continuous_verify_speaker else "OFF"
        print(
            f"Owner: {self.args.speaker_name}; speaker verification: {verification_status}; "
            f"threshold: {self.args.speaker_threshold:.2f}"
        )
        print("🎤 Đang nghe liên tục, không dùng wake word. Ctrl+C để thoát.", flush=True)
        previous_speech = False
        loud_barge_in_seconds = 0.0
        barge_in_triggered_for_segment = False
        segment_started_during_tts = False
        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=window_size, device=self.args.input_device, callback=microphone_callback,
            ):
                while True:
                    microphone_block, microphone_rms = audio_queue.get()
                    vad.accept_waveform(microphone_block)
                    speech = vad.is_speech_detected()
                    if speech and not previous_speech:
                        segment_started_during_tts = self._tts_playing.is_set()
                        print("\n🟢 Đang phát hiện giọng nói...", flush=True)
                        if self.args.audio_debug and self._tts_playing.is_set():
                            print(
                                "🔎 audio-debug VAD-onset-during-tts "
                                f"mic_rms={microphone_rms:.5f} "
                                f"barge_threshold={self.args.barge_in_rms:.5f}",
                                flush=True,
                            )
                    if speech and self._tts_playing.is_set() and self.args.barge_in:
                        # VAD alone is intentionally not enough: brief leakage,
                        # keyboard noise, and headset artefacts can all be speech-like.
                        # Require a sustained, clearly audible microphone level.
                        if microphone_rms >= self.args.barge_in_rms:
                            loud_barge_in_seconds += window_size / config.SAMPLE_RATE
                        if (
                            not barge_in_triggered_for_segment
                            and loud_barge_in_seconds >= self.args.barge_in_min_active_seconds
                        ):
                            if self.args.audio_debug:
                                print(
                                    "\n🔎 audio-debug Barge-in-confirmed "
                                    f"mic_rms={microphone_rms:.5f} "
                                    f"loud_for={loud_barge_in_seconds:.2f}s "
                                    f"threshold={self.args.barge_in_rms:.5f} "
                                    f"{self._last_audio_metrics}",
                                    flush=True,
                                )
                            self.interrupt_tts_for_barge_in()
                            barge_in_triggered_for_segment = True
                    elif previous_speech and not speech:
                        print("\n🔴 Đã phát hiện khoảng lặng; đang chốt đoạn...", flush=True)
                    if not speech:
                        loud_barge_in_seconds = 0.0
                    previous_speech = speech
                    while not vad.empty():
                        segment = vad.front
                        samples = np.ascontiguousarray(segment.samples, dtype=np.float32).copy()
                        vad.pop()
                        if segment_started_during_tts and not barge_in_triggered_for_segment:
                            print(
                                "\n🔇 Bỏ segment phát sinh trong lúc TTS: "
                                "chưa đạt điều kiện Barge-in.",
                                flush=True,
                            )
                            segment_started_during_tts = False
                            barge_in_triggered_for_segment = False
                            continue
                        try:
                            segment_queue.put_nowait(samples)
                        except queue.Full:
                            print("\n⚠️ Máy đang bận; bỏ segment cũ để giữ câu mới.", flush=True)
                            try:
                                segment_queue.get_nowait()
                                segment_queue.task_done()
                                segment_queue.put_nowait(samples)
                            except queue.Empty:
                                pass
                        segment_started_during_tts = False
                        barge_in_triggered_for_segment = False
        except KeyboardInterrupt:
            print("\nĐang dừng continuous owner mode...", flush=True)
        finally:
            try:
                segment_queue.put_nowait(None)
            except queue.Full:
                pass


def parse_args() -> argparse.Namespace:
    def device_identifier(value: str) -> int | str:
        """Preserve numeric sounddevice IDs while also accepting device names."""
        try:
            return int(value)
        except ValueError:
            return value

    parser = argparse.ArgumentParser(description="Mic -> (wake word | continuous VAD) -> owner verification -> STT -> LLM -> TTS")
    parser.add_argument("--input-device", type=device_identifier, default=None, help="Tên hoặc ID microphone (mặc định: microphone hệ thống)")
    parser.add_argument("--output-device", type=device_identifier, default=None, help="Tên hoặc ID loa (mặc định: loa hệ thống)")
    parser.add_argument("--activation-mode", choices=("wake", "continuous"), default="wake", help="wake: chờ wake word; continuous: VAD liên tục, không dùng wake word")
    parser.add_argument(
        "--continuous-verify-speaker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chỉ ở continuous mode: xác minh owner trước LLM/TTS (mặc định: bật)",
    )
    parser.add_argument(
        "--aec",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Khử echo TTS cục bộ trước VAD (mặc định: tắt; chỉ bật khi đã hiệu chỉnh theo thiết bị)",
    )
    parser.add_argument("--aec-delay-ms", type=float, default=100.0, help="Độ trễ loa tới mic cho AEC, tính bằng ms (mặc định: 100)")
    parser.add_argument("--aec-echo-correlation", type=float, default=0.35, help="Tương quan tối thiểu để bỏ echo loa (mặc định: 0.35)")
    parser.add_argument(
        "--barge-in",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cả hai mode: ngắt TTS khi VAD nhận giọng nói đủ to và đủ lâu (mặc định: bật)",
    )
    parser.add_argument(
        "--barge-in-rms",
        type=float,
        default=0.006,
        help="Mức RMS microphone tối thiểu để Barge-in được tính là rõ (mặc định: 0.006)",
    )
    parser.add_argument(
        "--barge-in-min-active-seconds",
        type=float,
        default=0.25,
        help="Thời lượng mic phải vượt ngưỡng RMS trong một lượt VAD trước khi ngắt TTS (mặc định: 0.25)",
    )
    parser.add_argument("--wake-model", type=Path, default=DEFAULT_WAKE_MODEL)
    parser.add_argument("--wake-threshold", type=float, default=config.settings.wake_threshold)
    parser.add_argument("--speaker-name", default=SPEAKER_NAME)
    parser.add_argument("--speaker-threshold", type=float, default=config.settings.speaker_threshold)
    parser.add_argument("--verify-seconds", type=float, default=SPEAKER_VERIFY_SECONDS)
    parser.add_argument("--audio-queue-seconds", type=float, default=10.0)
    parser.add_argument("--segment-queue-size", type=int, default=8)
    parser.add_argument(
        "--audio-debug",
        action="store_true",
        help="In mức tín hiệu mic/AEC khi TTS đang phát; không ghi hoặc lưu audio",
    )
    parser.add_argument(
        "--audio-debug-interval",
        type=float,
        default=0.25,
        help="Khoảng cách log audio debug, tính bằng giây (mặc định: 0.25)",
    )
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
