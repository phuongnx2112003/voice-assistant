from pathlib import Path
import numpy as np
import sherpa_onnx
class VietnameseSTT:
    '''Uses the official 30M INT8 Vietnamese ZipFormer as an offline recognizer.
    We re-decode the growing utterance for partial UI: simulated streaming, CPU-only.'''
    def __init__(self, model_dir, num_threads=2):
        d=Path(model_dir)
        needed=['encoder.int8.onnx','decoder.onnx','joiner.int8.onnx','tokens.txt']
        missing=[x for x in needed if not (d/x).exists()]
        if missing: raise FileNotFoundError(f'STT model incomplete in {d}; missing {missing}. Run scripts/download_sherpa_models.sh')
        self.r=sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d/'encoder.int8.onnx'), decoder=str(d/'decoder.onnx'),
            joiner=str(d/'joiner.int8.onnx'), tokens=str(d/'tokens.txt'),
            num_threads=num_threads, provider='cpu')
    def transcribe(self, audio, sr=16000):
        if len(audio)==0: return ''
        s=self.r.create_stream(); s.accept_waveform(sr, np.asarray(audio,dtype=np.float32)); self.r.decode_stream(s)
        return s.result.text.strip()
