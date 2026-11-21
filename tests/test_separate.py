import numpy as np
import pytest
from conftest import RATE, BandSeparator

from moviefilter import separate as sp


def tone(f, n, a=0.3):
    return (a * np.sin(2 * np.pi * f * np.arange(n) / RATE)).astype(np.float32)


def test_crossfade_edges_and_middle():
    o = np.ones((1000, 2), np.float32)
    r = np.zeros((1000, 2), np.float32)
    out = sp.crossfade_replace(o, r, 100)
    assert out[0, 0] == pytest.approx(1) and out[-1, 0] == pytest.approx(1)
    assert np.all(out[100:900] == 0)


def test_gain_envelope():
    g = sp.gain_envelope(1000, 50, -30)
    assert g[0] == pytest.approx(1) and g[500] == pytest.approx(10 ** (-30 / 20))


def test_plan_windows_merges_and_caps():
    T = sp.Target
    ts = [T("a", 10, 11, 10, 11), T("b", 13, 14, 13, 14), T("c", 100, 101, 100, 101)]
    w = sp.plan_windows(ts, 200)
    assert [(x.start, x.end, len(x.targets)) for x in w] == [(6, 18, 2), (96, 105, 1)]
    many = [T(str(i), i * 5.0, i * 5.0 + 0.5, i * 5.0, i * 5.0 + 0.5) for i in range(1, 60)]
    assert all(x.end - x.start <= sp.MAX_WINDOW for x in sp.plan_windows(many, 400))


def test_snap_range_extends_to_real_word():
    n = RATE * 3
    v = np.zeros(n, np.float32)
    v[int(1.0 * RATE):int(1.6 * RATE)] = tone(3000, int(0.6 * RATE))      # real word 1.0-1.6 s
    s, e = sp.snap_range(v, RATE, 1.2, 1.35)                               # whisper said 1.2-1.35
    assert s == pytest.approx(1.0, abs=0.02) and e == pytest.approx(1.6, abs=0.02)
    s, e = sp.snap_range(v, RATE, 1.2, 1.35, max_ext=0.1)                  # limited extension
    assert s == pytest.approx(1.1, abs=0.02)


def test_snap_range_silence_is_noop():
    s, e = sp.snap_range(np.zeros(RATE, np.float32), RATE, 0.2, 0.4)
    assert (s, e) == (0.2, 0.4)


def make_window(word_at=(2.0, 2.5), dur=5.0, channels=2):
    n = int(dur * RATE)
    bg = tone(200, n, 0.2)
    voice = np.zeros(n, np.float32)
    a, b = int(word_at[0] * RATE), int(word_at[1] * RATE)
    voice[a:b] = tone(3000, b - a)
    if channels == 2:
        return np.stack([bg + voice, bg + voice], 1), voice, bg
    z = np.zeros(n, np.float32)
    return np.stack([bg, bg, voice, z, bg, bg], 1), voice, bg


def band(x, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return spec[(f > lo) & (f < hi)].sum()


def stream(ch):
    return {"rate": RATE, "channels": ch, "layout": "stereo" if ch == 2 else "5.1(side)"}


def test_cleaner_removes_voice_keeps_background():
    audio, voice, bg = make_window()
    c = sp.Cleaner(BandSeparator(), stream(2), [(0, 1)], [0, 1], verify="fast")
    # whisper timestamps deliberately late/short: snapping must still cover the whole word
    t = sp.Target("x", 2.15, 2.4, 2.15, 2.4)
    out, ranges, rep = c.process(0.0, audio, [t])
    a, b = int(2.0 * RATE), int(2.5 * RATE)
    assert band(out[a:b, 0], 2500, 3500) < 0.01 * band(audio[a:b, 0], 2500, 3500)
    assert band(out[a:b, 0], 150, 250) == pytest.approx(band(bg[a:b], 150, 250), rel=0.02)
    assert np.allclose(out[: int(1.5 * RATE)], audio[: int(1.5 * RATE)])          # untouched before
    assert rep[0]["status"] == "clean" and rep[0]["residual_db"] < -30


def test_cleaner_surround_only_touches_groups():
    audio, voice, bg = make_window(channels=6)
    c = sp.Cleaner(BandSeparator(), stream(6), [(0, 1), (2,)], [2], verify="off")
    out, _, _ = c.process(0.0, audio, [sp.Target("x", 2.0, 2.5, 2.0, 2.5)])
    a, b = int(2.0 * RATE), int(2.5 * RATE)
    assert band(out[a:b, 2], 2500, 3500) < 0.01 * band(audio[a:b, 2], 2500, 3500)
    assert np.array_equal(out[:, 3], audio[:, 3]) and np.array_equal(out[:, 4], audio[:, 4])


class LeakySeparator(BandSeparator):
    """First separation leaves half the voice in the background; later ones are perfect."""

    def separate(self, audio, rate):
        v, g = super().separate(audio, rate)
        if self.calls == 1:
            return v * 0.5, g + v * 0.5
        return v, g


class HopelessSeparator(BandSeparator):
    def separate(self, audio, rate):
        v, g = super().separate(audio, rate)
        return v * 0.0, g + v                # never removes anything


def test_escalates_to_second_pass():
    audio, _, _ = make_window()
    c = sp.Cleaner(LeakySeparator(), stream(2), [(0, 1)], [0, 1], verify="fast")
    out, _, rep = c.process(0.0, audio, [sp.Target("x", 2.0, 2.5, 2.0, 2.5)])
    assert rep[0]["status"] == "clean-pass2"
    a, b = int(2.05 * RATE), int(2.45 * RATE)
    assert band(out[a:b, 0], 2500, 3500) < 0.02 * band(audio[a:b, 0], 2500, 3500)


def test_ducks_as_last_resort():
    audio, _, _ = make_window()

    class Asr:
        def transcribe(self, *a, **k):
            return [{"word": "fuck", "start": 2.1, "end": 2.4, "prob": 0.9}], "en"

    from moviefilter.wordlist import Matcher
    c = sp.Cleaner(HopelessSeparator(), stream(2), [(0, 1)], [0, 1], transcriber=Asr(), matcher=Matcher(),
                   verify="full")
    out, _, rep = c.process(0.0, audio, [sp.Target("x", 2.0, 2.5, 2.0, 2.5)])
    assert rep[0]["status"] == "ducked" and rep[0]["heard_after"]
    a, b = int(2.1 * RATE), int(2.4 * RATE)
    assert np.abs(out[a:b]).max() < 0.05 * np.abs(audio[a:b]).max()


def test_whisper_hallucination_is_not_ducked():
    """Voice measured gone, Whisper still 'hears' the word from context: flag it, don't duck it."""
    audio, _, bg = make_window()

    class Asr:
        def transcribe(self, *a, **k):
            assert k.get("prompt") is False          # verification must not use the swear-word prompt
            return [{"word": "fuck", "start": 2.1, "end": 2.4, "prob": 0.9}], "en"

    from moviefilter.wordlist import Matcher
    c = sp.Cleaner(BandSeparator(), stream(2), [(0, 1)], [0, 1], transcriber=Asr(), matcher=Matcher(),
                   verify="full")
    out, _, rep = c.process(0.0, audio, [sp.Target("x", 2.0, 2.5, 2.0, 2.5)])
    assert rep[0]["status"] == "check"
    a, b = int(2.1 * RATE), int(2.4 * RATE)
    assert band(out[a:b, 0], 150, 250) == pytest.approx(band(bg[a:b], 150, 250), rel=0.05)   # not ducked


def test_verification_ignores_unsure_or_neighbouring_words():
    audio, _, _ = make_window()

    class Asr:
        def transcribe(self, *a, **k):
            return [{"word": "shit", "start": 2.1, "end": 2.4, "prob": 0.3},      # unsure
                    {"word": "fuck", "start": 2.45, "end": 2.9, "prob": 0.9}], "en"  # mostly outside

    from moviefilter.wordlist import Matcher
    c = sp.Cleaner(BandSeparator(), stream(2), [(0, 1)], [0, 1], transcriber=Asr(), matcher=Matcher(),
                   verify="full")
    _, _, rep = c.process(0.0, audio, [sp.Target("x", 2.0, 2.5, 2.0, 2.5)])
    assert rep[0]["status"] == "clean" and not rep[0]["heard_after"]


def test_resample_roundtrip():
    x = tone(440, RATE)[:, None]
    y = sp.resample(sp.resample(x, 48000, 44100), 44100, 48000)
    assert np.corrcoef(x[1000:-1000, 0], y[1000:len(x) - 1000, 0])[0, 1] > 0.999
