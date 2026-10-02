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

Nếu không muốn dùng wake word, enroll owner trước rồi chạy chế độ nghe liên tục:

```bash
python -m voice_assistant.cli.enroll_speaker
python -m voice_assistant.cli.voice_cli --activation-mode continuous
```

Khi enroll, hãy nói liên tục trong mỗi mẫu. Các mẫu quá lệch so với mẫu đã
nhận sẽ bị yêu cầu thu lại; chạy enrollment lại sẽ thay thế embeddings cũ.

Muốn nghe thử và kiểm tra file thu trước, dùng:

```bash
python -m voice_assistant.cli.record_speaker_samples
```

Các WAV được lưu vào `samples/speaker/<timestamp>/`.

Luồng continuous là `microphone callback → Silero VAD → STT → speaker verification → LLM → TTS`.
Mỗi câu được VAD chốt sau khoảng lặng được xác minh lại; chỉ giọng đã enroll mới được gửi sang LLM/TTS.

Có thể tắt speaker verification cho continuous mode (mọi giọng nói được VAD nhận sẽ gọi LLM/TTS):

```bash
python -m voice_assistant.cli.voice_cli --activation-mode continuous --no-continuous-verify-speaker
```

Mặc định, cả continuous và wake mode đều bật **barge-in**: khi VAD nghe người
dùng nói lúc TTS đang phát, audio bị ngắt ngay và câu mới được xử lý. Để tắt
cho mode đang chạy, thêm `--no-barge-in`. Nên dùng tai nghe; project chưa có
acoustic echo cancellation nên loa ngoài có thể bị mic nhận nhầm là giọng nói mới.

Wake mode cũng mặc định bật barge-in. Sau khi một wake session đã được xác
minh, nói trong lúc TTS đang phát sẽ ngắt TTS và trở thành lượt tiếp theo mà
không cần nói lại wake word. Tắt bằng `--no-barge-in`.

## Commands

```bash
python -m voice_assistant.cli.enroll_speaker
python -m voice_assistant.cli.verify_speaker
python -m voice_assistant.cli.voice_cli
python -m voice_assistant.cli.voice_cli --activation-mode continuous
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
