from pathlib import Path
import wave
import numpy as np
import sherpa_onnx


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/stt/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"


# =========================================================
# CREATE RECOGNIZER
# =========================================================

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{MODEL_DIR}/encoder.int8.onnx",
    decoder=f"{MODEL_DIR}/decoder.onnx",
    joiner=f"{MODEL_DIR}/joiner.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",

    num_threads=2,

    sample_rate=16000,
    feature_dim=80,

    decoding_method="greedy_search",

    provider="cpu",
)


# =========================================================
# READ WAV
# =========================================================

def read_wave(filename):

    with wave.open(filename) as f:

        sample_rate = f.getframerate()

        num_samples = f.getnframes()

        samples = f.readframes(num_samples)

        samples = np.frombuffer(
            samples,
            dtype=np.int16,
        )

        samples = samples.astype(np.float32) / 32768

        return samples, sample_rate


# =========================================================
# TEST
# =========================================================

wav_file = f"{MODEL_DIR}/test_wavs/0.wav"

samples, sample_rate = read_wave(wav_file)


stream = recognizer.create_stream()

stream.accept_waveform(
    sample_rate,
    samples,
)

recognizer.decode_stream(stream)


print("Kết quả:")
print(stream.result.text)