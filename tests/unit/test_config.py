import unittest
from voice_assistant import config

class ConfigTests(unittest.TestCase):
    def test_project_paths_and_settings_are_valid(self):
        self.assertTrue(config.ROOT.is_dir())
        self.assertEqual(config.SAMPLE_RATE, 16000)
        self.assertGreater(config.settings.vad_threshold, 0)
        self.assertGreater(config.settings.max_utterance_seconds, 0)
