import torch
from pyannote.audio import Pipeline


class Diarizer:

    def __init__(
        self,
        hf_token: str,
        use_gpu: bool = False,
        model_name: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.pipeline.to(device)
        self.device = device

    def diarize(
        self,
        audio_path: str,
        min_segment_duration: float = 0.5,
    ) -> list[dict]:
        diarization = self.pipeline(audio_path)
        segments: list[dict] = []
        for turn, speaker in diarization.speaker_diarization:
            duration = turn.end - turn.start
            if duration < min_segment_duration:
                continue
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker_label": str(speaker),
                }
            )
        return segments