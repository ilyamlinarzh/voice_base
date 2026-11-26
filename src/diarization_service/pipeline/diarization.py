import torch
import soundfile as sf
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
        data, samplerate = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

        diarization = self.pipeline({"waveform": waveform, "sample_rate": samplerate})

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

        self._mark_overlap(segments)
        return segments

    @staticmethod
    def _mark_overlap(segments: list[dict]) -> None:
        for seg in segments:
            seg["is_overlap"] = False
            seg["overlap_intervals"] = []

        n = len(segments)
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = segments[i], segments[j]
                if s1["speaker_label"] == s2["speaker_label"]:
                    continue
                ov_start = max(s1["start"], s2["start"])
                ov_end = min(s1["end"], s2["end"])
                if ov_start < ov_end:
                    s1["is_overlap"] = True
                    s2["is_overlap"] = True
                    s1["overlap_intervals"].append((ov_start, ov_end))
                    s2["overlap_intervals"].append((ov_start, ov_end))

        for seg in segments:
            seg["overlap_duration"] = sum(
                e - s for s, e in seg["overlap_intervals"]
            )

    @staticmethod
    def clean_subsegments(seg: dict) -> list[tuple[float, float]]:
        if not seg.get("overlap_intervals"):
            return [(seg["start"], seg["end"])]

        ovs = sorted(seg["overlap_intervals"])

        clean = []
        cursor = seg["start"]
        for ov_start, ov_end in ovs:
            if ov_start > cursor:
                clean.append((cursor, ov_start))
            cursor = max(cursor, ov_end)
        if cursor < seg["end"]:
            clean.append((cursor, seg["end"]))

        return clean