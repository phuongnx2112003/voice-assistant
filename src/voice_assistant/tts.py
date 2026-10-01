"""Offline Vietnamese Piper/VITS text to speech."""
from pathlib import Path
import sherpa_onnx

class VietnameseTTS:
    def __init__(self, model_dir, num_threads=2):
        directory = Path(model_dir)
        required = ["vi_VN-vais1000-medium.onnx", "tokens.txt", "espeak-ng-data"]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"TTS model incomplete in {directory}; missing {missing}")
        config = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(directory / "vi_VN-vais1000-medium.onnx"),
                data_dir=str(directory / "espeak-ng-data"),
                tokens=str(directory / "tokens.txt"),
            ), num_threads=num_threads, provider="cpu"))
        if not config.validate():
            raise RuntimeError("Invalid TTS configuration")
        self._tts = sherpa_onnx.OfflineTts(config)
        self._generation = sherpa_onnx.GenerationConfig()
        self._generation.sid, self._generation.speed, self._generation.silence_scale = 0, 1.0, 0.2

    def synthesize(self, text):
        return self._tts.generate(text.strip(), self._generation)
