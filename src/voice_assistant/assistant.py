import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .audio import Microphone
from .endpoint import EndpointDetector
from .wakeword import WakeWord
from .vad import SileroVAD
from .stt import VietnameseSTT

class VoiceAssistant:
    def __init__(self, cfg, device=None):
        self.cfg=cfg
        self.mic=Microphone(cfg.SAMPLE_RATE,cfg.BLOCK_SAMPLES,device)
        self.wake=WakeWord(cfg.WAKE_MODEL,cfg.WAKE_THRESHOLD)
        self.vad=SileroVAD(cfg.VAD_MODEL, sample_rate=cfg.SAMPLE_RATE)
        self.stt=None
        self._decoder=ThreadPoolExecutor(max_workers=1, thread_name_prefix='stt')
        self._stt_load=None
        self._cooldown_until=0.0

    def _start_stt_load(self):
        if self.stt is None and self._stt_load is None:
            self._stt_load=self._decoder.submit(
                VietnameseSTT,self.cfg.STT_DIR,self.cfg.STT_THREADS)

    def _poll_stt_load(self):
        if self.stt is None and self._stt_load is not None and self._stt_load.done():
            self.stt=self._stt_load.result()

    def _stt(self):
        if self.stt is None:
            self._start_stt_load()
            self.stt=self._stt_load.result()
        return self.stt

    def run(self):
        try:
            self.mic.start(); print('WAIT_WAKE: nói wake word của model (vd. Hey Phượng). Ctrl+C để thoát.')
            while True:
                x=self.mic.read()
                score,_=self.wake.process(x)
                sys.stdout.write(f'\rwake={score:.3f}   '); sys.stdout.flush()
                if time.monotonic() >= self._cooldown_until and score >= self.cfg.WAKE_THRESHOLD:
                    print('\n🔥 WAKE'); self._start_stt_load(); self._listen()
                    self.wake.reset()
                    self._cooldown_until=time.monotonic()+self.cfg.WAKE_COOLDOWN_SECONDS
                    print('\nWAIT_WAKE')
        except KeyboardInterrupt: print('\nBye')
        finally:
            self.mic.stop()
            self._decoder.shutdown(wait=True)

    def _listen(self):
        self.vad.reset()
        chunks=[]; last_decode=0.0; partial=None
        endpoint=EndpointDetector(self.cfg.INITIAL_SPEECH_TIMEOUT_SECONDS,self.cfg.SILENCE_SECONDS)
        while endpoint.elapsed < self.cfg.MAX_UTTERANCE_SECONDS:
            x=self.mic.read(); chunks.append(x)
            speech,_=self.vad.is_speech(x)
            endpoint.observe(len(x)/self.cfg.SAMPLE_RATE,speech)
            now=time.monotonic()
            self._poll_stt_load()
            if partial is not None and partial.done():
                text=partial.result()
                sys.stdout.write('\r🎤 '+text+' '*10); sys.stdout.flush(); last_decode=now
                partial=None
            if self.stt is not None and partial is None and now-last_decode >= self.cfg.PARTIAL_DECODE_INTERVAL_SECONDS:
                audio=np.concatenate(chunks)
                partial=self._decoder.submit(self.stt.transcribe,audio,self.cfg.SAMPLE_RATE)
                last_decode=now
            if endpoint.should_stop(): break
        if partial is not None:
            partial.result()
        audio=np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float32)
        print('\nFINAL:',self._stt().transcribe(audio,self.cfg.SAMPLE_RATE))
