import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RATE = 48000
VOICE_HZ, BG_HZ = 3000.0, 200.0


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run real Whisper/Demucs integration tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs --slow (downloads models, takes minutes)")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


# ----------------------------------------------------------------- fake engines
class BandSeparator:
    """'Voice' = everything above 1 kHz, 'background' = below. Exact for the synthetic tones."""

    def __init__(self):
        self.calls = 0

    def separate(self, audio, rate):
        self.calls += 1
        n = len(audio)
        spec = np.fft.rfft(audio, axis=0)
        f = np.fft.rfftfreq(n, 1 / rate)
        hi = spec.copy()
        hi[f < 1000] = 0
        voice = np.fft.irfft(hi, n, axis=0).astype(np.float32)
        return voice, (audio - voice).astype(np.float32)


class ToneTranscriber:
    """Returns scripted words, but only where the 3 kHz 'voice' tone is actually audible."""

    def __init__(self, script, shift=0.0, scale=1.0):
        self.script = [dict(w, start=w["start"] * scale + shift, end=w["end"] * scale + shift) for w in script]
        self.model_name = "fake"
        self.language = "en"
        self.key = "fake"
        self.calls = 0

    def transcribe(self, audio16k, offset=0.0, vad=None, language=None, prompt=True):
        self.calls += 1
        dur = len(audio16k) / 16000
        out = []
        for w in self.script:
            if w["start"] < offset or w["end"] > offset + dur:
                continue
            a, b = int((w["start"] - offset) * 16000), int((w["end"] - offset) * 16000)
            seg = audio16k[a:b]
            if len(seg) < 160:
                continue
            spec = np.abs(np.fft.rfft(seg))
            f = np.fft.rfftfreq(len(seg), 1 / 16000)
            if spec[(f > 2500) & (f < 3500)].sum() / max(spec.sum(), 1e-9) > 0.2:
                out.append(dict(w))
        return out, "en"


SENTENCE = "the old harbor ships came in early this morning before".split()
SCRIPT = (
    [{"word": w, "start": 2.0 + i * 0.5, "end": 2.0 + i * 0.5 + 0.35, "prob": 0.95} for i, w in enumerate(SENTENCE)]
    + [{"word": "fuck", "start": 10.0, "end": 10.4, "prob": 0.9},
       {"word": "this", "start": 10.5, "end": 10.85, "prob": 0.95},
       {"word": "oh", "start": 19.5, "end": 19.85, "prob": 0.95},
       {"word": "shit", "start": 20.0, "end": 20.4, "prob": 0.92},
       {"word": "damn", "start": 24.0, "end": 24.4, "prob": 0.9},
       {"word": "bitch", "start": 27.0, "end": 27.4, "prob": 0.9}]
)


def tone_audio(script, dur, shift=0.0, scale=1.0, channels=2):
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    bg = (0.2 * np.sin(2 * np.pi * BG_HZ * t)).astype(np.float32)
    voice = np.zeros(n, np.float32)
    for w in script:
        a, b = int((w["start"] * scale + shift) * RATE), int((w["end"] * scale + shift) * RATE)
        voice[a:b] = 0.3 * np.sin(2 * np.pi * VOICE_HZ * t[a:b])
    if channels == 2:
        return np.stack([bg + voice, bg + voice], 1), voice, bg
    z = np.zeros(n, np.float32)            # 5.1(side): voice only in the center
    return np.stack([bg, bg, voice + 0.1 * bg, z, bg * 0.5, bg * 0.5], 1), voice, bg


def write_movie(path, audio, layout="stereo", codec="flac"):
    path = Path(path)
    raw = path.with_suffix(".f32")
    raw.write_bytes(audio.astype(np.float32).tobytes())
    dur = len(audio) / RATE
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=64x36:r=10:d={dur:.3f}",
                    "-f", "f32le", "-ar", str(RATE), "-ch_layout", layout, "-i", str(raw),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40",
                    "-c:a", codec, "-metadata:s:a:0", "language=eng", str(path)], check=True)
    raw.unlink()
    return path


def decode(path, order, channels):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{order}", "-f", "f32le",
                          "-ac", str(channels), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVIEFILTER_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def fake_engines(script, shift=0.0, scale=1.0):
    from moviefilter.pipeline import Engines
    eng = Engines(model="fake")
    eng._asr = ToneTranscriber(script, shift, scale)
    eng._sep = BandSeparator()
    return eng


def dump(obj):
    return json.dumps(obj, indent=1)
