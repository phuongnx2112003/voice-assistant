import unittest

from voice_assistant.endpoint import EndpointDetector


class EndpointDetectorTests(unittest.TestCase):
    def test_does_not_endpoint_during_initial_silence(self):
        detector=EndpointDetector(initial_speech_timeout=4.0,silence_timeout=1.0)
        for _ in range(12):
            detector.observe(0.08,False)
        self.assertFalse(detector.should_stop())

    def test_initial_timeout_ends_empty_turn(self):
        detector=EndpointDetector(initial_speech_timeout=0.24,silence_timeout=1.0)
        for _ in range(3):
            detector.observe(0.08,False)
        self.assertTrue(detector.should_stop())

    def test_silence_endpoints_only_after_speech(self):
        detector=EndpointDetector(initial_speech_timeout=4.0,silence_timeout=0.24)
        detector.observe(0.08,True)
        detector.observe(0.08,False)
        detector.observe(0.08,False)
        self.assertFalse(detector.should_stop())
        detector.observe(0.08,False)
        self.assertTrue(detector.should_stop())


if __name__ == '__main__':
    unittest.main()
