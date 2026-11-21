"""Voice removal around cuss words.

For each window of audio around one or more cuss words:
  1. Demucs splits every channel group (front L/R, center, optionally surrounds) into
     voice + background.
  2. The cut boundaries are snapped to where the voice actually starts/stops (Whisper's
     word timestamps are often 50-300 ms off).
  3. Inside each range the original is replaced by the background (short crossfade).
  4. Verification: the cleaned window is separated again to measure leftover voice, and
     (optionally) re-transcribed with Whisper to make sure the word is gone.
     Failures escalate: pass 2 = separate the cleaned audio again and use that
     background; pass 3 = duck the range 30 dB as a last resort.
"""
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .util import log

FADE = 0.015            # crossfade at splice edges (s)
FRAME = 0.01            # energy frame for boundary snapping (s)
MAX_EXT = 0.35          # how far snapping may extend a range past the word timestamps (s)
SNAP_REL = 0.07         # voice counts as "still talking" above ~-23 dB of the word's level
PAD_IN, PAD_OUT = 0.04, 0.06      # small safety margin after snapping
CONTEXT = 4.0           # audio context given to Demucs on each side of a range (s)
MAX_WINDOW = 90.0       # longest window processed at once (s)
RESIDUAL_DB = -16.0     # leftover voice louder than this (relative to original) = fail
DUCK_DB = -30.0
VERIFY_CONTEXT = 1.5    # context around a range when re-separating to verify (s)
CONFIRMED_CLEAN_DB = -30.0  # leftover voice this low = clean, even if Whisper claims to hear the word
HEARD_MIN_PROB = 0.5    # ignore low-confidence "heard" words during verification


def resample(x, src, dst):
    if src == dst:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(src), int(dst))
    return resample_poly(x, int(dst) // g, int(src) // g, axis=0).astype(np.float32)


def fit_len(x, n):
    if len(x) >= n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


class DemucsSeparator:
    """Voice/background separation with Demucs (loaded once, reused)."""

    def __init__(self, model="htdemucs", device="auto", shifts=1, overlap=0.25):
        self.name, self.shifts, self.overlap = model, shifts, overlap
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from demucs.pretrained import get_model
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            log(f"  loading Demucs model {self.name} on {self.device} ...")
            m = get_model(self.name)
            m.eval()
            self._model = m.to(self.device)
        return self._model

    def separate(self, audio, rate):
        """audio: float32 [n, 1|2]. Returns (voice, background), same shape and rate."""
        import torch
        from demucs.apply import apply_model
        model = self._load()
        n, ch = audio.shape
        x = np.repeat(audio, 2, axis=1) if ch == 1 else audio
        xr = resample(x, rate, model.samplerate)
        if len(xr) < model.samplerate:          # demucs wants at least ~1 s
            xr = fit_len(xr, model.samplerate)
        wav = torch.from_numpy(np.ascontiguousarray(xr.T)).to(self.device)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std()
        if float(std) < 1e-7:
            return np.zeros_like(audio), audio.copy()
        with torch.no_grad():
            out = apply_model(model, ((wav - mean) / std)[None], device=self.device, shifts=self.shifts,
                              split=True, overlap=self.overlap, progress=False)[0]
        out = out * std + mean
        vi = model.sources.index("vocals")
        voc = out[vi].cpu().numpy().T
        bg = (out.sum(0) - out[vi]).cpu().numpy().T
        voc = fit_len(resample(voc, model.samplerate, rate), n)
        bg = fit_len(resample(bg, model.samplerate, rate), n)
        if ch == 1:
            voc, bg = voc.mean(1, keepdims=True), bg.mean(1, keepdims=True)
        return voc.astype(np.float32), bg.astype(np.float32)


# ------------------------------------------------------------------ pure helpers
def crossfade_replace(orig, rep, fade_n):
    """Replace orig with rep, linearly crossfading over fade_n samples at both edges."""
    n = min(len(orig), len(rep))
    orig, rep = orig[:n], rep[:n]
    w = np.ones(n, dtype=np.float32)
    f = min(fade_n, n // 2)
    if f > 0:
        w[:f] = np.linspace(0, 1, f, dtype=np.float32)
        w[n - f:] = np.linspace(1, 0, f, dtype=np.float32)
    shape = (n,) + (1,) * (orig.ndim - 1)
    w = w.reshape(shape)
    return orig * (1 - w) + rep * w


def gain_envelope(n, fade_n, db):
    g = 10 ** (db / 20)
    w = np.full(n, g, dtype=np.float32)
    f = min(fade_n, n // 2)
    if f > 0:
        w[:f] = np.linspace(1, g, f, dtype=np.float32)
        w[n - f:] = np.linspace(g, 1, f, dtype=np.float32)
    return w


def snap_range(voice, rate, core_start, core_end, max_ext=MAX_EXT):
    """Extend [core_start, core_end] (seconds, window-relative) to cover the voiced burst.

    Walks outward from the word's timestamps while the separated voice is still loud,
    so late/early Whisper timestamps don't leave half a syllable behind.
    """
    hop = max(1, int(FRAME * rate))
    nfr = len(voice) // hop
    if nfr == 0:
        return core_start, core_end
    frames = voice[: nfr * hop].reshape(nfr, hop)
    e = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    a = min(max(0, int(core_start / FRAME)), nfr - 1)
    b = min(max(a + 1, int(math.ceil(core_end / FRAME))), nfr)
    core = e[a:b]
    if core.size == 0 or core.max() <= 1e-6:
        return core_start, core_end
    thr = max(1e-4, SNAP_REL * float(np.percentile(core, 90)))
    lim = int(max_ext / FRAME)
    s = a
    while s > 0 and a - s < lim and e[s - 1] > thr:
        s -= 1
    t = b
    while t < nfr and t - b < lim and e[t] > thr:
        t += 1
    return s * FRAME, t * FRAME


@dataclass
class Target:
    id: str
    start: float          # seconds, this copy's clock (already padded a little)
    end: float
    core_start: float     # raw word timestamps
    core_end: float
    level: str = "strong"


@dataclass
class WindowPlan:
    start: float
    end: float
    targets: list = field(default_factory=list)


def plan_windows(targets, total, context=CONTEXT, max_window=MAX_WINDOW):
    wins = []
    for t in sorted(targets, key=lambda t: t.start):
        s, e = max(0.0, t.start - context), min(total, t.end + context)
        if wins and s <= wins[-1].end and e - wins[-1].start <= max_window:
            wins[-1].end = max(wins[-1].end, e)
            wins[-1].targets.append(t)
        else:
            wins.append(WindowPlan(s, e, [t]))
    return wins


# ------------------------------------------------------------------ the cleaner
class Cleaner:
    def __init__(self, separator, stream, groups, dialog_channels, transcriber=None, matcher=None,
                 verify="full", language=None):
        self.sep = separator
        self.rate = stream["rate"]
        self.groups = groups
        self.dialog = dialog_channels
        self.asr = transcriber
        self.matcher = matcher
        self.verify = verify            # off | fast | full
        self.language = language

    def _separate_groups(self, audio):
        voice = np.zeros_like(audio)
        bg = audio.copy()
        for g in self.groups:
            v, b = self.sep.separate(audio[:, list(g)], self.rate)
            voice[:, list(g)] = v
            bg[:, list(g)] = b
        return voice, bg

    def _separate_regions(self, audio, ranges, ctx=VERIFY_CONTEXT):
        """Separate only around the given sample ranges (+context). Cheaper than the whole window."""
        voice = np.zeros_like(audio)
        bg = audio.copy()
        c = int(ctx * self.rate)
        crops = []
        for a, b in sorted(ranges):
            a0, b0 = max(0, a - c), min(len(audio), b + c)
            if crops and a0 <= crops[-1][1]:
                crops[-1][1] = max(crops[-1][1], b0)
            else:
                crops.append([a0, b0])
        for a0, b0 in crops:
            v, g = self._separate_groups(audio[a0:b0])
            voice[a0:b0] = v
            bg[a0:b0] = g
        return voice, bg

    def _voice_mono(self, voice):
        return voice[:, self.dialog].mean(axis=1)

    def _splice(self, base, source, ranges):
        out = base.copy()
        fade = int(FADE * self.rate)
        chans = sorted({c for g in self.groups for c in g})
        for a, b in ranges:
            a0, b0 = max(0, a - fade), min(len(out), b + fade)
            out[a0:b0, chans] = crossfade_replace(base[a0:b0, chans], source[a0:b0, chans], fade)
        return out

    def _residual_db(self, before_voice, after_voice, a, b, mix=None):
        """Leftover voice vs original voice in dB, or None if Demucs found (almost) no voice there,
        in which case the measurement says nothing and Whisper's verdict decides."""
        r0 = rms(before_voice[a:b])
        r1 = rms(after_voice[a:b])
        if r0 < 1e-6 or (mix is not None and r0 < 0.1 * rms(mix[a:b])):
            return None
        return 20 * math.log10(max(r1, 1e-9) / r0)

    def _heard(self, audio, win_t0, ranges_sec):
        if self.verify != "full" or self.asr is None or self.matcher is None:
            return [False] * len(ranges_sec)
        mono16 = resample(audio.mean(axis=1), self.rate, 16000)
        words, _ = self.asr.transcribe(mono16, offset=win_t0, vad=False, language=self.language, prompt=False)
        dets = [d for d in self.matcher.find(words) if d.prob >= HEARD_MIN_PROB]
        res = []
        for s, e in ranges_sec:
            # the heard word must sit mostly inside the cleaned range, not just touch it
            res.append(any(min(d.end, e) - max(d.start, s) >= 0.5 * max(d.end - d.start, 0.05) for d in dets))
        return res

    @staticmethod
    def _failed(residual, heard):
        """Leftover voice is the primary signal; Whisper alone can hallucinate from context."""
        if residual is None:
            return heard
        return residual > RESIDUAL_DB or (heard and residual > CONFIRMED_CLEAN_DB)

    def process(self, win_t0, audio, targets):
        """Clean one window. Returns (cleaned_audio, ranges_in_samples, reports)."""
        rate = self.rate
        voice, bg = self._separate_groups(audio)
        vmono = self._voice_mono(voice)
        ranges = []
        for t in targets:
            cs, ce = t.core_start - win_t0, t.core_end - win_t0
            ss, se = snap_range(vmono, rate, cs, ce)
            # snapped to the real voice: small margins are enough (less collateral on neighbours)
            s = min(cs, ss) - PAD_IN
            e = max(ce, se) + PAD_OUT
            ranges.append([max(0, int(s * rate)), min(len(audio), int(e * rate))])
        merged = []
        for a, b in sorted(ranges):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        cleaned = self._splice(audio, bg, merged)
        status = ["clean"] * len(merged)
        residual = [None] * len(merged)
        heard = [False] * len(merged)
        if self.verify != "off":
            v2, bg2 = self._separate_regions(cleaned, merged)
            v2m = self._voice_mono(v2)
            mix_d = audio[:, self.dialog].mean(axis=1)
            residual = [self._residual_db(vmono, v2m, a, b, mix_d) for a, b in merged]
            heard = self._heard(cleaned, win_t0, [(a / rate + win_t0, b / rate + win_t0) for a, b in merged])
            bad = [i for i in range(len(merged)) if self._failed(residual[i], heard[i])]
            if bad:
                cleaned = self._splice(cleaned, bg2, [merged[i] for i in bad])
                v3, _ = self._separate_regions(cleaned, [merged[i] for i in bad])
                v3m = self._voice_mono(v3)
                heard3 = self._heard(cleaned, win_t0, [(merged[i][0] / rate + win_t0, merged[i][1] / rate + win_t0) for i in bad])
                fade = int(FADE * rate)
                for j, i in enumerate(bad):
                    a, b = merged[i]
                    residual[i] = self._residual_db(vmono, v3m, a, b, mix_d)
                    heard[i] = heard3[j]
                    if self._failed(residual[i], heard[i]):
                        a0, b0 = max(0, a - fade), min(len(cleaned), b + fade)
                        cleaned[a0:b0] *= gain_envelope(b0 - a0, fade, DUCK_DB)[:, None]
                        status[i] = "ducked"
                    else:
                        status[i] = "clean-pass2"
            for i in range(len(merged)):
                if status[i] != "ducked" and heard[i]:
                    status[i] = "check"          # voice measured gone, but Whisper still "hears" it
        reports = [{"start": round(a / rate + win_t0, 3), "end": round(b / rate + win_t0, 3),
                    "status": status[i],
                    "residual_db": None if residual[i] is None else round(residual[i], 1),
                    "heard_after": bool(heard[i])} for i, (a, b) in enumerate(merged)]
        return cleaned, merged, reports


# ------------------------------------------------------------------ stem transcription helper
def voice_16k(separator, mono16):
    """Isolate the voice from 16 kHz mono audio (used to re-transcribe hard passages)."""
    v, _ = separator.separate(mono16[:, None].astype(np.float32), 16000)
    return v[:, 0]
