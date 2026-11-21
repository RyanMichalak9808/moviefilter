"""Whisper transcription with word timestamps, overlapping chunks and an on-disk cache."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .util import Progress, log, read_json, write_json

CHUNK = 600.0     # seconds each chunk "owns"
OVERLAP = 5.0     # extra audio on each side so words at a boundary are never cut in half
EN_PROMPT = "Damn it. Oh shit. What the fuck? Son of a bitch. Hell no, you asshole."


def default_model(language):
    return "small.en" if language == "en" else "small"


class Transcriber:
    """Thin wrapper around faster-whisper. Swap with a fake in tests."""

    def __init__(self, model="small.en", language="en", device="auto", compute_type="auto",
                 vad=True, beam_size=5):
        self.model_name = model
        self.language = None if language in (None, "auto") else language
        if model.endswith(".en"):
            self.language = "en"
        self.device = device
        self.compute_type = compute_type
        self.vad = vad
        self.beam_size = beam_size
        self._model = None

    @property
    def key(self):
        return f"{self.model_name}-{self.language or 'auto'}-{'vad' if self.vad else 'novad'}-b{self.beam_size}"

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            ct = self.compute_type
            if ct == "auto":
                ct = "int8"
                try:
                    import torch
                    if self.device in ("auto", "cuda") and torch.cuda.is_available():
                        ct = "float16"
                except ImportError:
                    pass
            log(f"  loading Whisper model {self.model_name} ({ct}) ...")
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=ct)
        return self._model

    def transcribe(self, audio16k, offset=0.0, vad=None, language=None, prompt=True):
        """audio16k: float32 mono @16 kHz. Returns (words, detected_language).

        prompt=False for verification: the swear-word prompt makes Whisper more willing to
        "hear" a cuss word from context alone, which is good for finding them, bad for checking.
        """
        if len(audio16k) < 1600:
            return [], language or self.language
        lang = language or self.language
        vad = self.vad if vad is None else vad
        kw = dict(word_timestamps=True, beam_size=self.beam_size, language=lang,
                  condition_on_previous_text=False, vad_filter=vad)
        if vad:
            kw["vad_parameters"] = dict(threshold=0.35, min_silence_duration_ms=400, speech_pad_ms=300)
        if lang == "en" and prompt:
            # A profane prompt stops Whisper from "cleaning up" or skipping swear words.
            kw["initial_prompt"] = EN_PROMPT
        segs, info = self.model.transcribe(np.ascontiguousarray(audio16k, dtype=np.float32), **kw)
        words = []
        for s in segs:
            for w in s.words or []:
                if not w.word.strip():
                    continue
                words.append({"start": round(offset + w.start, 3), "end": round(offset + w.end, 3),
                              "word": w.word.strip(), "prob": round(float(w.probability), 3)})
        return words, info.language


def transcribe_audio(audio, transcriber, cache_dir, label="transcribe"):
    """Transcribe a whole Audio16k in overlapping chunks. Resumable: each chunk is cached."""
    cache_dir = Path(cache_dir) / transcriber.key
    cache_dir.mkdir(parents=True, exist_ok=True)
    dur = audio.duration
    n = max(1, int(np.ceil(dur / CHUNK)))
    meta = read_json(cache_dir / "meta.json", {})
    language = transcriber.language or meta.get("language")
    words = []
    prog = Progress(label, n)
    for k in range(n):
        own0, own1 = k * CHUNK, min(dur, (k + 1) * CHUNK)
        f = cache_dir / f"chunk_{k:05d}.json"
        cached = read_json(f)
        if cached is None:
            a, b = max(0.0, own0 - OVERLAP), min(dur, own1 + OVERLAP)
            ws, lang = transcriber.transcribe(audio.read(a, b), offset=a, language=language)
            if language is None and ws:
                language = lang
                write_json(cache_dir / "meta.json", {"language": language})
                log(f"  detected language: {language}")
            cached = ws
            write_json(f, cached)
        # keep words whose midpoint falls in the span this chunk owns (dedupes the overlap)
        words += [w for w in cached if own0 <= (w["start"] + w["end"]) / 2 < own1 or (k == n - 1 and (w["start"] + w["end"]) / 2 >= own1)]
        prog.update()
    prog.done()
    return words, language or "en"


def transcribe_span(audio, transcriber, cache_dir, start, end, language=None, vad=None):
    """Transcribe an arbitrary span, cached by (start, end, settings)."""
    start, end = max(0.0, start), min(audio.duration, end)
    h = hashlib.sha1(f"{start:.2f}-{end:.2f}-{transcriber.key}-{language}-{vad}".encode()).hexdigest()[:16]
    f = Path(cache_dir) / "spans" / f"{h}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    cached = read_json(f)
    if cached is None:
        cached, _ = transcriber.transcribe(audio.read(start, end), offset=start, language=language, vad=vad)
        write_json(f, cached)
    return cached


def words_to_text(words):
    return " ".join(w["word"] for w in words)


def save_transcript(words, path):
    Path(path).write_text(json.dumps(words))
