# Troubleshooting

## No microphone / PortAudio error
Run `python scripts/list_audio_devices.py`, then `python -m voice_assistant --input-device N`.

## Wake model missing
Train/export a custom openWakeWord ONNX model and put it at `models/wakeword/hey_phuong.onnx`.

## STT model missing
Run `./scripts/download_sherpa_models.sh`.

## STT is slow
Set `STT_THREADS` in `config.py` to 1, 2, or 4 and benchmark. The current partial transcript path re-decodes the growing utterance, so CPU use rises with utterance length.

## Cuts the end too early / too late
Adjust `SILENCE_SECONDS` and `VAD_THRESHOLD` in `.env`; the runtime uses Silero VAD.
