class EndpointDetector:
    """Endpoint only after speech has started, with a separate start timeout."""

    def __init__(self, initial_speech_timeout, silence_timeout):
        self.initial_speech_timeout = initial_speech_timeout
        self.silence_timeout = silence_timeout
        self.elapsed = 0.0
        self.silence = 0.0
        self.speech_started = False

    def observe(self, duration, is_speech):
        self.elapsed += duration
        if is_speech:
            self.speech_started = True
            self.silence = 0.0
        elif self.speech_started:
            self.silence += duration

    def should_stop(self):
        if not self.speech_started:
            return self.elapsed >= self.initial_speech_timeout
        return self.silence >= self.silence_timeout
