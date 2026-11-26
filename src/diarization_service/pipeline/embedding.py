import numpy as np
import torch
from pyannote.audio import Model, Inference
from pyannote.core import Segment
import soundfile as sf


class EmbeddingExtractor:

    def __init__(
        self,
        hf_token: str,
        use_gpu: bool = False,
        model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
    ) -> None:
        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        model = Model.from_pretrained(
            model_name,
            token=hf_token,
        )
        model.to(device)
        model.eval()
        self.inference = Inference(model, window="whole", device=device)
        self.device = device

    def extract(
        self,
        audio_path: str,
        start: float,
        end: float,
    ) -> np.ndarray:
        data, samplerate = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        duration = waveform.shape[-1] / samplerate

        start = max(0.0, float(start))
        end = min(float(end), duration)

        if end <= start:
            raise ValueError(
                f"Invalid segment after clipping: start={start}, end={end}, "
                f"duration={duration}"
            )

        audio_dict = {"waveform": waveform, "sample_rate": samplerate}
        segment = Segment(start=start, end=end)
        embedding = self.inference.crop(audio_dict, segment)

        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        if arr.shape != (256,):
            raise ValueError(f"Unexpected embedding shape: {arr.shape}, expected (256,)")
        return arr
    
    def extract_from_array(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        embedding = self.inference({
            "waveform": waveform_tensor,
            "sample_rate": sample_rate,
        })
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        if arr.shape != (256,):
            raise ValueError(
                f"Unexpected embedding shape: {arr.shape}, expected (256,)"
            )
        return arr