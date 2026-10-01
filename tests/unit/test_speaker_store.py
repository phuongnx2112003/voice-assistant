import tempfile
import unittest
from pathlib import Path
import numpy as np
from voice_assistant import speaker_store

class SpeakerStoreTests(unittest.TestCase):
    def test_replace_and_load_centroid(self):
        original_database = speaker_store.SPEAKER_DATABASE
        with tempfile.TemporaryDirectory() as directory:
            try:
                speaker_store.SPEAKER_DATABASE = Path(directory) / "speakers.sqlite3"
                speaker_store.replace_embeddings("test-owner", [np.array([1, 0], dtype=np.float32), np.array([1, 0], dtype=np.float32)])
                np.testing.assert_allclose(speaker_store.load_centroid("test-owner"), np.array([1, 0], dtype=np.float32))
            finally:
                speaker_store.SPEAKER_DATABASE = original_database
