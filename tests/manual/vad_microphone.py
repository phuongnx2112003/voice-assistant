from pathlib import Path

import sounddevice as sd
import sherpa_onnx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 16000
VAD_MODEL = str(ROOT / "models/vad/silero_vad.onnx")


# ============================================================
# VAD CONFIG
# ============================================================

vad_config = sherpa_onnx.VadModelConfig()

vad_config.silero_vad.model = VAD_MODEL
vad_config.silero_vad.threshold = 0.5
vad_config.silero_vad.min_silence_duration = 0.8
vad_config.silero_vad.min_speech_duration = 0.25
vad_config.sample_rate = SAMPLE_RATE

# QUAN TRỌNG
WINDOW_SIZE = vad_config.silero_vad.window_size

print("VAD window size:", WINDOW_SIZE)


vad = sherpa_onnx.VoiceActivityDetector(
    vad_config,
    buffer_size_in_seconds=30,
)


# ============================================================
# MICROPHONE
# ============================================================

print("Microphone:")
print(sd.query_devices())

print("\nDefault input:", sd.default.device[0])

print("\n🎤 Đang nghe...")


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=WINDOW_SIZE,
) as mic:

    previous_speech_state = False

    while True:

        samples, overflowed = mic.read(WINDOW_SIZE)

        samples = samples.reshape(-1)
        samples = np.copy(samples)

        # Debug microphone volume
        rms = np.sqrt(np.mean(samples ** 2))
        peak = np.max(np.abs(samples))

        print(
            f"\rRMS={rms:.5f} "
            f"PEAK={peak:.5f}",
            end="",
            flush=True,
        )

        # Feed đúng 1 window vào VAD
        vad.accept_waveform(samples)

        # VAD hiện đang thấy speech?
        speech_detected = vad.is_speech_detected()

        if speech_detected and not previous_speech_state:
            print("\n🟢 BẮT ĐẦU NÓI")

        if not speech_detected and previous_speech_state:
            print("\n🔴 CÓ THỂ ĐÃ KẾT THÚC SPEECH")

        previous_speech_state = speech_detected

        # Có segment hoàn chỉnh
        while not vad.empty():

            segment = vad.front
            vad.pop()

            duration = len(segment.samples) / SAMPLE_RATE

            print(
                f"\n✅ SPEECH SEGMENT: {duration:.2f}s"
            )