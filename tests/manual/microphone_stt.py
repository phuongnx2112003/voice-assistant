"""Vietnamese microphone STT with VAD and non-blocking audio capture.

The recognizer in this project is an *offline* ZipFormer recognizer. It emits
one final result after VAD sees a pause; audio capture must therefore not wait
for decoding a previous utterance.
"""

import queue
import threading

import numpy as np
import sherpa_onnx
import sounddevice as sd


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 16_000
MODEL_DIR = ROOT / "models/stt/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
VAD_MODEL = str(ROOT / "models/vad/silero_vad.onnx")

# A 512-sample Silero window is 32 ms. Keep enough microphone buffering for
# slow CPUs without letting old audio build up indefinitely.
AUDIO_QUEUE_SECONDS = 10
SEGMENT_QUEUE_SIZE = 8


recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{MODEL_DIR}/encoder.int8.onnx",
    decoder=f"{MODEL_DIR}/decoder.onnx",
    joiner=f"{MODEL_DIR}/joiner.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    num_threads=2,
    sample_rate=SAMPLE_RATE,
    feature_dim=80,
    decoding_method="greedy_search",
    provider="cpu",
)

vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model = VAD_MODEL
# A lower threshold is less likely to discard quiet initial syllables.
vad_config.silero_vad.threshold = 0.35
vad_config.silero_vad.min_speech_duration = 0.15
vad_config.silero_vad.min_silence_duration = 0.6
# Split a continuous monologue so it can still be transcribed before it ends.
vad_config.silero_vad.max_speech_duration = 12.0
vad_config.sample_rate = SAMPLE_RATE

WINDOW_SIZE = vad_config.silero_vad.window_size
vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)


def recognize(samples: np.ndarray) -> str:
    """Decode one completed VAD segment. Called only by the decoder thread."""
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


audio_queue: queue.Queue[np.ndarray] = queue.Queue(
    maxsize=int(AUDIO_QUEUE_SECONDS * SAMPLE_RATE / WINDOW_SIZE)
)
segment_queue: queue.Queue[np.ndarray | None] = queue.Queue(
    maxsize=SEGMENT_QUEUE_SIZE
)
stop_event = threading.Event()
overflow_count = 0
overflow_lock = threading.Lock()


def microphone_callback(indata, frames, time_info, status):
    """Run on sounddevice's callback thread: copy audio and return immediately."""
    del frames, time_info
    global overflow_count
    if status:
        with overflow_lock:
            overflow_count += 1
    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        # Never block the callback. Drop the oldest block and retain live audio.
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Empty:
            pass


def decoder_worker():
    while True:
        samples = segment_queue.get()
        try:
            if samples is None:
                return

            duration = len(samples) / SAMPLE_RATE
            print(f"\n🧠 Đang nhận dạng đoạn {duration:.2f}s...", flush=True)
            text = recognize(samples)
            if text:
                print(f"Bạn: {text}", flush=True)
            else:
                print("⚠️ STT không nhận dạng được đoạn này.", flush=True)
        finally:
            segment_queue.task_done()


def main():
    global overflow_count
    worker = threading.Thread(target=decoder_worker, name="stt-decoder", daemon=True)
    worker.start()

    print(f"VAD window size: {WINDOW_SIZE}")
    print("🎤 Đang nghe... Kết quả hiện sau khoảng 0.6 giây im lặng.")

    previous_speech_state = False
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=WINDOW_SIZE,
            callback=microphone_callback,
        ):
            while not stop_event.is_set():
                samples = audio_queue.get()
                vad.accept_waveform(samples)

                speech_detected = vad.is_speech_detected()
                if speech_detected and not previous_speech_state:
                    print("\n🟢 Bạn đang nói...", flush=True)
                elif not speech_detected and previous_speech_state:
                    print("\n🔴 Đã nhận thấy khoảng lặng.", flush=True)
                previous_speech_state = speech_detected

                with overflow_lock:
                    lost_blocks, overflow_count = overflow_count, 0
                if lost_blocks:
                    print(f"\n⚠️ Audio overflow ({lost_blocks} block).", flush=True)

                while not vad.empty():
                    segment = vad.front
                    vad.pop()
                    # Copy before the VAD reuses its internal storage.
                    try:
                        segment_queue.put_nowait(np.array(segment.samples, dtype=np.float32))
                    except queue.Full:
                        # Prefer the latest speech over stale queued speech.
                        segment_queue.get_nowait()
                        segment_queue.task_done()
                        segment_queue.put_nowait(np.array(segment.samples, dtype=np.float32))
                        print("\n⚠️ Máy đang bận nhận dạng; đã bỏ một đoạn cũ.", flush=True)
    except KeyboardInterrupt:
        print("\nĐang dừng...", flush=True)
    finally:
        stop_event.set()
        # This script is interactive; don't keep it alive waiting for old jobs.
        try:
            segment_queue.put_nowait(None)
        except queue.Full:
            pass


if __name__ == "__main__":
    main()
