# Training the custom “Hey Phượng” wake word

The ZIP does **not** pretend to contain a trained `hey_phuong.onnx`: a reliable custom model must be trained/calibrated. openWakeWord's upstream repository includes a training workflow/Colab for custom wake words.

Recommended dataset plan:
- Positive: many speakers saying “Hey Phượng”, varied speed/pitch/distance.
- Hard negatives: “Phượng”, “Phương”, “Hey”, “Hey Siri”, “Phượng ơi”, similar Vietnamese syllables.
- Background negatives: conversation, TV, keyboard, fans, street/room noise.
- Augment positives with noise/reverb, but keep a clean held-out real recording set.

Export/copy the trained ONNX model to:
`models/wakeword/hey_phuong.onnx`

Then tune `WAKE_THRESHOLD` in `config.py`. Measure false accepts per hour and false rejects on a held-out test set; do not assume 0.5 is optimal.

Upstream: https://github.com/dscripka/openWakeWord
