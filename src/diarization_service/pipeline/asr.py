from faster_whisper import WhisperModel


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        use_gpu: bool = False,
        compute_type: str = "int8",
    ) -> None:
        device = "cuda" if use_gpu else "cpu"
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "ru",
    ) -> list[dict]:
        segments, _info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        words: list[dict] = []
        for segment in segments:
            if segment.words is None:
                continue
            for word in segment.words:
                text = word.word.strip()
                if not text:
                    continue
                words.append(
                    {
                        "text": text,
                        "start": float(word.start),
                        "end": float(word.end),
                        "probability": float(word.probability),
                    }
                )
        return words