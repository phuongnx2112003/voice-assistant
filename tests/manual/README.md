# Manual checks

These scripts can require a microphone, speaker, model files, or `.env` LLM credentials. They are intentionally named without the `test_` prefix so `pytest` will not execute them automatically.

Run from the repository root, for example:

```bash
python tests/manual/stt_model_check.py
python tests/manual/vad_microphone.py
python tests/manual/tts_llm_cli.py
```
