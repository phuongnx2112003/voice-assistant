import unittest

from voice_assistant.cli.voice_cli import text_for_speech


class TextForSpeechTests(unittest.TestCase):
    def test_removes_markdown_but_keeps_readable_content(self):
        source = "# **Xin chào**\n\n- Xem [trang chủ](https://example.com)\n- `lệnh` ~~cũ~~"
        self.assertEqual(text_for_speech(source), "Xin chào Xem trang chủ lệnh cũ")

    def test_removes_horizontal_rule_and_escaped_markers(self):
        self.assertEqual(text_for_speech("---\nNội dung \\*không đọc sao"), "Nội dung không đọc sao")
