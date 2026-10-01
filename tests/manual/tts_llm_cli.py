from pathlib import Path

import os
import re
import queue
import threading

import sounddevice as sd
import sherpa_onnx
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL_DIR = ROOT / "models/tts"


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Thiếu biến môi trường {name}. Hãy thêm nó vào file .env.")
    return value


# LLM settings are deliberately kept out of source control.
OPENAI_API_KEY = required_env("OPENAI_API_KEY")
OPENAI_BASE_URL = required_env("OPENAI_BASE_URL")
LLM_MODEL = required_env("OPENAI_MODEL")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# ============================================================
# SHERPA-ONNX TTS
# ============================================================

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{MODEL_DIR}/vi_VN-vais1000-medium.onnx",
            data_dir=f"{MODEL_DIR}/espeak-ng-data",
            tokens=f"{MODEL_DIR}/tokens.txt",
        ),
        num_threads=2,
        provider="cpu",
    ),
    max_num_sentences=1,
)

if not tts_config.validate():
    raise RuntimeError("TTS config không hợp lệ")

tts = sherpa_onnx.OfflineTts(tts_config)

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.silence_scale = 0.2


# ============================================================
# QUEUE
#
# LLM thread:
#       text -> sentence_queue
#
# TTS generation thread:
#       sentence_queue -> sherpa -> audio_queue
#
# Audio playback thread:
#       audio_queue -> speaker
# ============================================================

sentence_queue = queue.Queue()
audio_queue = queue.Queue()


def tts_worker():
    """
    Tổng hợp text thành audio trên một thread riêng.

    Playback không chạy ở đây, nên TTS có thể chuẩn bị câu tiếp theo
    trong lúc câu hiện tại đang phát qua loa.
    """

    while True:

        text = sentence_queue.get()

        if text is None:
            sentence_queue.task_done()
            break

        text = text.strip()

        if not text:
            sentence_queue.task_done()
            continue

        print(f"\n[TTS] {text}")

        try:
            audio = tts.generate(
                text,
                gen_config,
            )

            if len(audio.samples) > 0:
                audio_queue.put((audio.samples, audio.sample_rate))

        except Exception as e:
            print("\nTTS ERROR:", e)

        finally:
            sentence_queue.task_done()


def playback_worker():
    """Phát các audio đã tổng hợp theo đúng thứ tự câu trả lời."""

    while True:
        item = audio_queue.get()

        if item is None:
            audio_queue.task_done()
            break

        samples, sample_rate = item

        try:
            sd.play(samples, samplerate=sample_rate, blocking=True)
        except Exception as e:
            print("\nPLAYBACK ERROR:", e)
        finally:
            audio_queue.task_done()


# ============================================================
# START TTS THREAD
# ============================================================

tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True,
)

playback_thread = threading.Thread(
    target=playback_worker,
    daemon=True,
)

tts_thread.start()
playback_thread.start()


# ============================================================
# SENTENCE DETECTION
# ============================================================

def extract_sentences(buffer):
    """
    Ví dụ:

        "Chào bạn. Tôi là Kira. Hôm"

    ->

        [
            "Chào bạn.",
            "Tôi là Kira."
        ]

    remaining:

        "Hôm"
    """

    sentences = []

    while True:

        match = re.search(
            r"(.+?[.!?…])(?:\s+|$)",
            buffer,
            flags=re.DOTALL,
        )

        if not match:
            break

        sentence = match.group(1).strip()

        if sentence:
            sentences.append(sentence)

        buffer = buffer[match.end():]

    return sentences, buffer


def text_for_speech(text):
    """Chuyển Markdown từ LLM thành văn bản tự nhiên trước khi đọc TTS."""

    # Giữ nhãn của link nhưng bỏ URL; ví dụ [OpenAI](https://openai.com) -> OpenAI.
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()

        # Bỏ đường kẻ ngang Markdown như "---".
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Bỏ ký hiệu heading, quote và bullet/list nhưng giữ nội dung của chúng.
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)

        # Bỏ ký hiệu định dạng: **đậm**, *nghiêng*, `code`, ~~gạch ngang~~.
        line = re.sub(r"[*_`~]", "", line)
        line = line.replace("\\", "")

        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


# ============================================================
# LLM STREAMING
# ============================================================

def ask_llm(user_text):

    print("\nKira: ", end="", flush=True)

    stream = client.responses.create(
        model=LLM_MODEL,

        instructions=(
            "Bạn là Kira, một trợ lý giọng nói thân thiện.\n\n"
            "QUY TẮC BẮT BUỘC VỀ NGÔN NGỮ:\n"
            "- Luôn trả lời hoàn toàn bằng tiếng Việt Unicode có đầy đủ dấu.\n"
            "- Tuyệt đối không viết tiếng Việt không dấu.\n"
            "- Ví dụ đúng: ‘Hoa Kỳ là một quốc gia rộng lớn.’\n"
            "- Ví dụ sai: ‘Hoa Ky la mot quoc gia rong lon.’\n"
            "- Trước khi trả lời, hãy tự kiểm tra và thêm dấu tiếng Việt cho mọi từ cần có dấu.\n\n"
            "ĐỊNH DẠNG CÂU TRẢ LỜI:\n"
            "- Bạn có thể dùng Markdown để người dùng dễ đọc, ví dụ tiêu đề, danh sách "
            "và chữ in đậm.\n"
            "- Trả lời tự nhiên, rõ ràng và ưu tiên các câu ngắn để dễ nghe."
        ),

        input=user_text,

        stream=True,
    )

    buffer = ""

    for event in stream:

        # text delta từ Responses API
        if event.type == "response.output_text.delta":

            token = event.delta

            # Hiển thị text ngay lập tức
            print(token, end="", flush=True)

            buffer += token

            # xem đã có câu hoàn chỉnh chưa
            sentences, buffer = extract_sentences(buffer)

            for sentence in sentences:

                # Làm sạch Markdown trước khi đưa sang TTS thread.
                speech_text = text_for_speech(sentence)
                if speech_text:
                    sentence_queue.put(speech_text)

    # phần text cuối cùng có thể không kết thúc bằng .
    speech_text = text_for_speech(buffer)
    if speech_text:
        sentence_queue.put(speech_text)

    print()


# ============================================================
# MAIN
# ============================================================

try:

    while True:

        user_text = input("\nBạn: ").strip()

        if not user_text:
            continue

        if user_text.lower() in {
            "exit",
            "quit",
            "thoát",
        }:
            break

        ask_llm(user_text)

        # Đợi cả phần tổng hợp lẫn phát audio xong trước câu hỏi tiếp theo.
        sentence_queue.join()
        audio_queue.join()


finally:

    sentence_queue.put(None)
    tts_thread.join()
    audio_queue.put(None)
    playback_thread.join()
