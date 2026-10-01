# Hey Phượng — Vietnamese Voice Assistant

A local CPU-first pipeline: microphone/PCM → wake word → Silero VAD → optional speaker verification → Vietnamese STT → optional LLM → Vietnamese TTS.

## Setup

```bash
./scripts/setup_ubuntu.sh
source .venv/bin/activate
./scripts/download_sherpa_models.sh
```

The custom wake word is not downloadable. Train `hey_kira.onnx` from `notebooks/train_kira_full.ipynb`, place it in `models/wakeword/`, then run:

```bash
python -m voice_assistant.cli.enroll_speaker
python -m voice_assistant.cli.voice_cli --wake-model models/wakeword/hey_kira.onnx
```

## Commands

```bash
python -m voice_assistant.cli.enroll_speaker
python -m voice_assistant.cli.verify_speaker
python -m voice_assistant.cli.voice_cli
python -m voice_assistant
```

Environment settings are documented in `.env.example`. Models are downloaded to `models/` by `scripts/download_sherpa_models.sh`.

## Development

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
python tests/manual/stt_model_check.py
```

Manual tests may use a microphone, local model or LLM credentials; they are intentionally excluded from pytest.
