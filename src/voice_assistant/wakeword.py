from pathlib import Path
import numpy as np
class WakeWord:
    def __init__(self, model_path, threshold=.5):
        p=Path(model_path)
        if not p.exists(): raise FileNotFoundError(f'Wake-word model not found: {p}\nSee docs/TRAIN_WAKEWORD.md')
        from openwakeword.model import Model
        self.model=Model(wakeword_models=[str(p)], inference_framework='onnx')
        self.threshold=threshold
    def process(self, audio_float):
        pcm=np.clip(audio_float*32767,-32768,32767).astype(np.int16)
        pred=self.model.predict(pcm)
        score=max((float(v) for v in pred.values()), default=0.0)
        return score, pred

    def reset(self):
        self.model.reset()
