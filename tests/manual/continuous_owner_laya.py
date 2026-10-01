"""Manual test for continuous-owner activation.

Pipeline: microphone callback -> VAD -> worker -> STT (diagnostic) -> owner
speaker verification -> Laya session intent. Wake word, LLM, and TTS are not
used. STT is intentionally printed even for a non-owner, but only an owner is
allowed to reach Laya.

Prerequisites:
    python -m voice_assistant.cli.enroll_speaker
    bash scripts/install_laya_cpu.sh

Run from the repository root:
    python tests/manual/continuous_owner_laya.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import sherpa_onnx
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from voice_assistant import config
from voice_assistant.speaker import SpeakerVerifier
from voice_assistant.speaker_store import SPEAKER_NAME
from voice_assistant.stt import VietnameseSTT

SAMPLE_RATE = config.SAMPLE_RATE
AUDIO_QUEUE_SECONDS = 10
SEGMENT_QUEUE_SIZE = 8
QUESTIONS = {
    "session_intent": {
        "type": "choice",
        "instructions": "Phân loại ý định điều khiển phiên của trợ lý bằng tiếng Việt.",
        "criteria": {
            "start_session": "Người dùng yêu cầu bật, khởi động, bắt đầu hoặc đánh thức trợ lý/phiên trò chuyện.",
            "end_session": "Người dùng yêu cầu tắt, dừng, thoát hoặc kết thúc trợ lý/phiên trò chuyện.",
            "normal_request": "Người dùng đặt câu hỏi hoặc yêu cầu trợ lý thực hiện việc bình thường, không yêu cầu mở hay đóng phiên.",
            "unknown": "Câu nói không rõ ý, không phải yêu cầu cho trợ lý, hoặc không thể phân loại an toàn.",
        },
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-device", help="Sounddevice input device name or index")
    parser.add_argument("--threads", type=int, default=config.settings.stt_threads)
    parser.add_argument("--laya-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--intent-threshold", type=float, default=0.80)
    parser.add_argument("--vad-threshold", type=float, default=0.25)
    parser.add_argument("--silence-seconds", type=float, default=0.45)
    parser.add_argument("--min-speech-seconds", type=float, default=0.10)
    return parser.parse_args()


def build_vad(args: argparse.Namespace) -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    """Use responsive VAD defaults for short commands and single words."""
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = str(config.VAD_MODEL)
    vad_config.silero_vad.threshold = args.vad_threshold
    vad_config.silero_vad.min_speech_duration = args.min_speech_seconds
    vad_config.silero_vad.min_silence_duration = args.silence_seconds
    vad_config.silero_vad.max_speech_duration = 15.0
    vad_config.sample_rate = SAMPLE_RATE
    detector = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
    return detector, vad_config.silero_vad.window_size


def load_laya(device: str):
    try:
        import laya
    except ModuleNotFoundError as error:
        raise RuntimeError("Chưa cài Laya. Chạy: bash scripts/install_laya_cpu.sh") from error
    print("Đang nạp Laya multilingual; lần đầu có thể mất vài phút...", flush=True)
    return laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)


def classify(agent, text: str) -> tuple[str, float, float]:
    started = time.perf_counter()
    result = agent.predict({"utterance": text}, QUESTIONS)
    elapsed_ms = (time.perf_counter() - started) * 1000
    answer = result["answers"]["session_intent"]
    confidence = float(answer.get("confidence", answer.get("answer_confidence", 0.0)))
    return str(answer["choice"]), confidence, elapsed_ms


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if not 0.0 <= args.intent_threshold <= 1.0:
        raise ValueError("--intent-threshold must be between 0 and 1")
    if not 0.0 < args.vad_threshold < 1.0:
        raise ValueError("--vad-threshold must be between 0 and 1")
    if args.silence_seconds <= 0 or args.min_speech_seconds <= 0:
        raise ValueError("VAD durations must be positive")

    speaker = SpeakerVerifier(
        config.SPEAKER_MODEL,
        SPEAKER_NAME,
        config.settings.speaker_threshold,
        sample_rate=SAMPLE_RATE,
    )
    stt = VietnameseSTT(config.STT_DIR, num_threads=args.threads)
    vad, window_size = build_vad(args)
    agent = load_laya(args.laya_device)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(
        maxsize=int(AUDIO_QUEUE_SECONDS * SAMPLE_RATE / window_size)
    )
    segment_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=SEGMENT_QUEUE_SIZE)
    session_active = False
    session_lock = threading.Lock()

    def process_segment(samples: np.ndarray) -> None:
        nonlocal session_active
        duration = samples.size / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        print(f"\n🎙️ VAD chốt đoạn: {duration:.2f}s (rms={rms:.4f}, peak={peak:.4f})", flush=True)
        text = stt.transcribe(samples, SAMPLE_RATE)
        if text:
            print(f"📝 STT: {text}", flush=True)
        else:
            print("⚠️ STT không tạo được text.", flush=True)

        try:
            verification = speaker.verify(samples)
        except ValueError as error:
            print(f"⚠️ Speaker embedding chưa đủ tin cậy: {error}", flush=True)
            return
        print(
            f"🔐 owner score={verification.centroid_score:.3f} "
            f"(threshold={verification.threshold:.2f})",
            flush=True,
        )
        if not verification.authorized:
            print("⛔ Non-owner: transcript chỉ để chẩn đoán; không gọi Laya.", flush=True)
            return
        if not text:
            print("⚠️ Owner hợp lệ nhưng STT rỗng; không gọi Laya.", flush=True)
            return

        intent, confidence, latency_ms = classify(agent, text)
        print(
            f"🧭 Laya: {intent} (confidence={confidence:.3f}, latency={latency_ms:.0f}ms)",
            flush=True,
        )
        with session_lock:
            if confidence < args.intent_threshold:
                print("⚠️ Confidence thấp: giữ nguyên session state.", flush=True)
            elif intent == "start_session" and not session_active:
                session_active = True
                print("✅ ACTION: OFF → ACTIVE", flush=True)
            elif intent == "end_session" and session_active:
                session_active = False
                print("✅ ACTION: ACTIVE → OFF", flush=True)
            elif intent == "normal_request" and session_active:
                print("✅ ACTION: ACTIVE; production có thể gửi text này sang LLM.", flush=True)
            elif intent == "normal_request":
                print("🔒 ACTION: OFF; chặn LLM vì session chưa được khởi động.", flush=True)
            else:
                state = "ACTIVE" if session_active else "OFF"
                print(f"ℹ️ ACTION: giữ session {state}.", flush=True)

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

    def microphone_callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"\n⚠️ Microphone: {status}", flush=True)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(indata[:, 0].copy())
                print("\n⚠️ Audio queue đầy; bỏ block cũ để giữ âm thanh mới.", flush=True)
            except queue.Empty:
                pass

    thread = threading.Thread(target=worker, name="continuous-owner-worker", daemon=True)
    thread.start()
    print("\n=== CONTINUOUS OWNER + LAYA TEST ===")
    print(f"Owner: {SPEAKER_NAME}; speaker threshold: {config.settings.speaker_threshold:.2f}")
    print(f"Intent threshold: {args.intent_threshold:.2f}; initial session: OFF")
    print(f"VAD: threshold={args.vad_threshold:.2f}, min-silence={args.silence_seconds:.2f}s")
    print("🎤 Đang nghe. Nói xong rồi im lặng khoảng {args.silence_seconds:.1f} giây để chốt STT. Ctrl+C để thoát.\n")

    previous_speech = False
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=window_size,
            device=args.input_device,
            callback=microphone_callback,
        ):
            while True:
                vad.accept_waveform(audio_queue.get())
                speech = vad.is_speech_detected()
                if speech and not previous_speech:
                    print("\n🟢 Đang phát hiện giọng nói...", flush=True)
                elif not speech and previous_speech:
                    print("\n🔴 Đã phát hiện khoảng lặng; đang chốt đoạn...", flush=True)
                previous_speech = speech

                while not vad.empty():
                    segment = vad.front
                    samples = np.ascontiguousarray(segment.samples, dtype=np.float32).copy()
                    vad.pop()
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
    except KeyboardInterrupt:
        print("\nĐang dừng continuous-owner test...", flush=True)
    finally:
        try:
            segment_queue.put_nowait(None)
        except queue.Full:
            pass


if __name__ == "__main__":
    main()
