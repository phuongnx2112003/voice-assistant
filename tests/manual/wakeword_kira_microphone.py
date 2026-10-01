from pathlib import Path

import sounddevice as sd
import numpy as np
from openwakeword.model import Model

ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
THRESHOLD = 0.5
model = Model(
    wakeword_models=[
        str(ROOT / "models/wakeword/hey_kira.onnx")
    ],
    inference_framework="onnx",
)

print("===================================")
print("🎤 Key Kira wake-word test")
print('Hãy nói: "HEY KIRA"')
print("Ctrl+C để thoát")
print("===================================")


def callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:, 0].astype(np.int16)

    predictions = model.predict(audio)

    for name, score in predictions.items():
        score = float(score)

        # Debug để xem model bắt đầu phản ứng
     
        print(f"{name}: {score:.6f}")

        if score >= THRESHOLD:
            print("\n🔥🔥🔥 HEY KIRA DETECTED 🔥🔥🔥")
            print(f"score = {score:.3f}\n")


try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=callback,
    ):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nStopped.")