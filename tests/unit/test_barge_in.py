import unittest
from unittest.mock import patch

from voice_assistant.cli.voice_cli import VoiceCli


class _Args:
    barge_in = True


class BargeInTests(unittest.TestCase):
    def test_stops_playback_when_tts_is_active(self):
        cli = VoiceCli.__new__(VoiceCli)
        cli.args = _Args()
        from threading import Event
        cli._tts_playing = Event()
        cli._tts_playing.set()
        with patch.object(cli, "_stop_tts_playback") as stop:
            self.assertTrue(cli.interrupt_tts_for_barge_in())
        stop.assert_called_once_with()
        self.assertFalse(cli._tts_playing.is_set())

    def test_does_nothing_when_not_playing(self):
        cli = VoiceCli.__new__(VoiceCli)
        cli.args = _Args()
        from threading import Event
        cli._tts_playing = Event()
        with patch.object(cli, "_stop_tts_playback") as stop:
            self.assertFalse(cli.interrupt_tts_for_barge_in())
        stop.assert_not_called()
