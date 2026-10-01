from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MODEL = str(ROOT / "models/wakeword/hey_kira.onnx")

# Kiểm tra ONNX
model = onnx.load(MODEL)
onnx.checker.check_model(model)
print("✓ ONNX model valid")

# Kiểm tra ONNX Runtime
session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

print("✓ ONNX Runtime load OK")

print("\nInputs:")
for x in session.get_inputs():
    print(" ", x.name, x.shape, x.type)

print("\nOutputs:")
for x in session.get_outputs():
    print(" ", x.name, x.shape, x.type)