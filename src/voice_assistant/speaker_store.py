"""Local SQLite storage and shared helpers for speaker verification.

The database contains biometric voice embeddings.  It is deliberately kept
inside this project and ignored by Git; it is not a shared or cloud database.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SAMPLE_RATE = 16_000
SPEAKER_MODEL = ROOT / "models/speaker/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
SPEAKER_DATABASE = Path(os.getenv("SPEAKER_DATABASE", ".data/speaker_embeddings.sqlite3"))
if not SPEAKER_DATABASE.is_absolute():
    SPEAKER_DATABASE = ROOT / SPEAKER_DATABASE
SPEAKER_NAME = os.getenv("SPEAKER_NAME", "owner").strip() or "owner"
SPEAKER_THRESHOLD = float(os.getenv("SPEAKER_THRESHOLD", "0.60"))
SPEAKER_ENROLL_SAMPLES = int(os.getenv("SPEAKER_ENROLL_SAMPLES", "5"))
SPEAKER_ENROLL_SECONDS = float(os.getenv("SPEAKER_ENROLL_SECONDS", "5"))
SPEAKER_VERIFY_SECONDS = float(os.getenv("SPEAKER_VERIFY_SECONDS", "4"))


def normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32).reshape(-1))
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Embedding không hợp lệ hoặc có năng lượng bằng 0")
    return embedding / norm


def _connect() -> sqlite3.Connection:
    SPEAKER_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SPEAKER_DATABASE)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS speaker_embeddings (
            id INTEGER PRIMARY KEY,
            speaker_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            sample_rate INTEGER NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_name "
        "ON speaker_embeddings(speaker_name)"
    )
    return connection


def replace_embeddings(speaker_name: str, embeddings: list[np.ndarray]) -> int:
    """Atomically replace one speaker's enrollment samples."""
    if not embeddings:
        raise ValueError("Cần ít nhất một embedding để enrollment")

    normalized = [normalize(item) for item in embeddings]
    dimensions = {item.size for item in normalized}
    if len(dimensions) != 1:
        raise ValueError("Các embedding có kích thước không đồng nhất")

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute("DELETE FROM speaker_embeddings WHERE speaker_name = ?", (speaker_name,))
        connection.executemany(
            """
            INSERT INTO speaker_embeddings
                (speaker_name, model_path, sample_rate, embedding_dim, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    speaker_name,
                    str(SPEAKER_MODEL.relative_to(ROOT)),
                    SAMPLE_RATE,
                    item.size,
                    item.tobytes(),
                    now,
                )
                for item in normalized
            ],
        )
    return len(normalized)


def load_centroid(speaker_name: str) -> np.ndarray:
    """Load all enrollment samples and return their normalized centroid."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT embedding_dim, embedding FROM speaker_embeddings
            WHERE speaker_name = ?
            ORDER BY id
            """,
            (speaker_name,),
        ).fetchall()

    if not rows:
        # Smooth upgrade from the original scripts, which stored one centroid
        # in owner_embedding.npy. New enrollments only use SQLite.
        legacy_file = ROOT / "owner_embedding.npy"
        if speaker_name == "owner" and legacy_file.is_file():
            try:
                replace_embeddings(speaker_name, [np.load(legacy_file, allow_pickle=False)])
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Không thể migrate {legacy_file.name}: {error}") from error
            return load_centroid(speaker_name)
        raise LookupError(
            f"Chưa có enrollment cho '{speaker_name}'. Hãy chạy: python -m voice_assistant.cli.enroll_speaker trước."
        )

    embeddings = []
    for dimension, blob in rows:
        embedding = np.frombuffer(blob, dtype=np.float32)
        if embedding.size != dimension:
            raise RuntimeError("Dữ liệu embedding trong database bị hỏng")
        embeddings.append(normalize(embedding))
    return normalize(np.mean(embeddings, axis=0))
