import queue
import numpy as np
import sounddevice as sd

class Microphone:
    def __init__(self, sample_rate=16000, block_samples=1280, device=None):
        self.sample_rate=sample_rate; self.block_samples=block_samples; self.device=device
        self.q=queue.Queue(maxsize=100); self.stream=None
    def _cb(self, indata, frames, time, status):
        if status: print(f'\n[audio] {status}')
        x=np.asarray(indata[:,0], dtype=np.float32).copy()
        try: self.q.put_nowait(x)
        except queue.Full:
            try: self.q.get_nowait(); self.q.put_nowait(x)
            except queue.Empty: pass
    def start(self):
        self.stream=sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
            blocksize=self.block_samples, device=self.device, callback=self._cb)
        self.stream.start()
    def read(self): return self.q.get()
    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()
