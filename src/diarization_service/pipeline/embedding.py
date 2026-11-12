import numpy as np
import torch
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.core import Segment


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

        segment = Segment(start=start, end=end)
        embedding = self.inference.crop(audio_path, segment)
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        if arr.shape != (256,):
            raise ValueError(
                f"Unexpected embedding shape: {arr.shape}, expected (256,)"
            )
        return arr