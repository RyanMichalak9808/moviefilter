"""Score a cleaned movie against the synthetic ground truth.

Metrics per cleaned range (computed on the dialogue channels):
  voice_left_db : energy of (output - true background) vs true voice   -> lower is better
  bg_kept_db    : energy of output vs true background                   -> ~0 dB is ideal (a mute would be << -20)
  bg_corr       : correlation of output with true background            -> ~1 is ideal
Plus: outside the ranges the clean track must match the original mix, and a fresh Whisper
pass over the clean track must not hear any strong cuss words.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moviefilter import media  # noqa: E402


def decode_track(path, order, channels, rate=48000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{order}", "-ac", str(channels),
                          "-ar", str(rate), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels)


def align(ref, x, max_lag=4800):
    """Lag (samples) that best aligns x to ref (codec delay check)."""
    a = ref[: 48000 * 20].mean(1)
    b = x[: 48000 * 20].mean(1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best, lag = -1, 0
    for L in range(-max_lag, max_lag + 1, 8):
        if L >= 0:
            c = np.dot(a[L:], b[: n - L])
        else:
            c = np.dot(a[: n + L], b[-L:])
        if c > best:
            best, lag = c, L
    return lag


def db(a, b):
    return 10 * np.log10((np.sum(a.astype(np.float64) ** 2) + 1e-12) / (np.sum(b.astype(np.float64) ** 2) + 1e-12))


def score(movie, cleaned, report):
    truth = json.loads(Path(str(movie) + ".truth.json").read_text())
    voice, _ = sf.read(truth["stems"]["voice"], dtype="float32")
    bg, _ = sf.read(truth["stems"]["bg"], dtype="float32")
    mix, _ = sf.read(truth["stems"]["mix"], dtype="float32")
    info = media.probe(cleaned)
    streams = media.audio_streams(info)
    clean = [s for s in streams if s["clean"]][-1]
    out = decode_track(cleaned, clean["order"], clean["channels"])
    orig = decode_track(movie, 0, clean["channels"])
    sync = media.best_lag(orig[: 48000 * 30].mean(1), out[: 48000 * 30].mean(1))   # clean vs original: must be 0
    lag = align(mix, orig)       # the source file's own codec delay vs the true stems
    # lag < 0: the decoded track is late by -lag samples (orig[i - lag] == mix[i])
    off_o, off_t = max(-lag, 0), max(lag, 0)
    n = min(len(out) - off_o, len(orig) - off_o, len(mix) - off_t) - 10
    sl = slice(off_o, off_o + n)
    tl = slice(off_t, off_t + n)
    out, orig = out[sl], orig[sl]
    voice, bg, mix = voice[tl], bg[tl], mix[tl]
    dia = [2] if clean["channels"] == 6 else [0, 1]

    rows = []
    mask = np.zeros(n, bool)
    for r in report["ranges"]:
        a, b = int(r["start"] * 48000) - off_o, int(r["end"] * 48000) - off_o
        a, b = max(0, a), min(n, b)
        mask[max(0, a - 2400): b + 2400] = True
        o, v, g = out[a:b, dia], voice[a:b, dia], bg[a:b, dia]
        vl = db(o - g, v)
        bk = db(o, g)
        corr = float(np.corrcoef(o.ravel(), g.ravel())[0, 1]) if g.std() > 0 else 0.0
        # voice leakage: the part of the output that is literally the original voice (projection),
        # separate from Demucs re-shaping the background a little
        vv = float(np.sum(v.astype(np.float64) ** 2))
        c = float(np.sum((o - g).astype(np.float64) * v)) / vv if vv > 0 else 0.0
        leak = 20 * np.log10(max(abs(c), 1e-6))
        rows.append({"start": r["start"], "status": r["status"], "voice_leak_db": round(float(leak), 1),
                     "residual_db": round(float(vl), 1),
                     "bg_kept_db": round(float(bk), 1), "bg_corr": round(corr, 3)})
    # outside the cleaned ranges the clean track should equal the original track
    diff_outside = db(out[~mask] - orig[~mask], orig[~mask]) if (~mask).any() else -99
    return {"sync_error_samples": sync, "source_lag": lag, "ranges": rows,
            "outside_diff_db": round(float(diff_outside), 1)}


def transcribe_clean(cleaned, model="small.en"):
    from moviefilter.asr import Transcriber
    from moviefilter.wordlist import Matcher
    info = media.probe(cleaned)
    clean = [s for s in media.audio_streams(info) if s["clean"]][-1]
    x = decode_track(cleaned, clean["order"], 1, 16000)[:, 0]
    words, _ = Transcriber(model, "en").transcribe(x, vad=False)
    dets = Matcher(levels=("strong", "slurs")).find(words)
    return words, [d.to_dict() for d in dets]


if __name__ == "__main__":
    movie, cleaned, report = sys.argv[1:4]
    res = score(movie, cleaned, json.loads(Path(report).read_text()))
    print(json.dumps(res, indent=1))
