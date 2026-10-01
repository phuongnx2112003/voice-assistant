# Architecture

`voice_assistant` is a reusable package under `src/`.

- `cli/`: operational command-line entry points.
- `service.py`: framework-independent request pipeline and structured `VoiceResult`.
- `vad.py`, `stt.py`, `speaker.py`, `tts.py`: model adapters.
- `speaker_store.py`: local SQLite biometric embedding store.

For an HTTP/WebSocket service, instantiate models once per worker. `VoiceService.process()` serializes calls because VAD/model sessions are stateful. A streaming API should own one VAD/session per connected client rather than sharing its VAD state across connections.

Input to `VoiceService.process()` must be mono float32 PCM at 16 kHz. It returns whether speech was detected, an optional speaker verification result, and final transcript. TTS remains an explicit `synthesize()` call.
