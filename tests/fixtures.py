"""Synthetic test movies with known ground truth (needs espeak-ng + ffmpeg).

Each movie = TTS dialogue (some lines profane) + synthetic music/ambience, rendered
as separate stems so tests can measure exactly how much voice was removed and how
much background survived.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 48000

SCRIPT = [
    ("en-us", "Welcome back to the harbor, the ships came in early this morning."),
    ("en-gb", "Get out of here, you son of a bitch."),
    ("en-us", "I told you the money was hidden under the old lighthouse."),
    ("en-us", "This is fucking ridiculous and you know it."),
    ("en-gb", "Captain Reynolds wants the cargo moved before sunset."),
    ("en-us", "Oh shit, the police are already at the gate."),
    ("en-gb", "We should take the northern road through the mountains."),
    ("en-us", "I don't give a damn about your excuses anymore."),
    ("en-us", "Listen carefully, the password is blue elephant seven."),
    ("en-gb", "You stupid bastard, you ruined everything we planned."),
    ("en-us", "The train to Chicago leaves at a quarter past nine."),
    ("en-us", "What the hell were you thinking last night?"),
    ("en-gb", "My grandmother used to bake bread every Sunday morning."),
    ("en-us", "Shut your mouth, asshole, before I lose my temper."),
    ("en-us", "The weather report says heavy rain is coming tomorrow."),
    ("en-gb", "Holy shit, look at the size of that wave."),
    ("en-us", "Bring the maps and the lantern to the second floor."),
    ("en-us", "Fuck this, I am going home right now."),
    ("en-gb", "The museum closes early on public holidays."),
    ("en-us", "That bitch stole my car and my wallet."),
    ("en-us", "Remember to water the tomato plants in the garden."),
    ("en-gb", "Goddamn it, the engine is overheating again."),
    ("en-us", "Thank you for coming, everybody, drive home safely."),
]


def tts(text, voice, out, speed=150):
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(out), text], check=True)
    x, sr = sf.read(out, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    from scipy.signal import resample_poly
    import math
    g = math.gcd(sr, RATE)
    return resample_poly(x, RATE // g, sr // g).astype(np.float32)


def music(n, seed=0):
    """Chord progression + soft pulse + pink-ish noise: stationary enough to judge, musical enough for Demucs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    chords = [(110, 138.6, 164.8), (98, 123.5, 146.8), (87.3, 110, 130.8), (130.8, 164.8, 196)]
    out = np.zeros(n, np.float32)
    seg = int(2.0 * RATE)
    for k in range(0, n, seg):
        f = chords[(k // seg) % 4]
        tt = t[k:k + seg]
        env = np.minimum(1, np.minimum((tt - tt[0]) * 8, (tt[-1] - tt + 1 / RATE) * 8))
        out[k:k + seg] += sum(np.sin(2 * np.pi * fr * tt) * (0.05 / (i + 1)) for i, fr in enumerate(f)) * env
    beat = (np.mod(t, 0.5) < 0.04).astype(np.float32) * rng.standard_normal(n).astype(np.float32) * 0.04
    pink = np.cumsum(rng.standard_normal(n)).astype(np.float32)
    pink = (pink - np.convolve(pink, np.ones(400) / 400, mode="same")) * 0.002
    return (out + beat + pink).astype(np.float32)


VOICES = ["en-us", "en-gb", "en-us+m3", "en-gb+f2", "en-us+f3", "en-gb-x-rp"]


def build_master(work, seed=0):
    """Render the dialogue + background stems once. Returns dict with arrays and line timings.

    seed varies voices, speaking speed, loudness, pauses and the music so repeated test runs
    are not all the same clip. seed 0 = the reference rendering.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    voice_parts, lines, pos = [], [], int(1.0 * RATE)
    for i, (v, text) in enumerate(SCRIPT):
        if seed:
            v = VOICES[rng.integers(len(VOICES))]
        speed = 150 if not seed else int(rng.integers(135, 180))
        x = tts(text, v, work / f"line{i}.wav", speed=speed)
        x = x / (np.abs(x).max() + 1e-9) * (0.5 if not seed else float(rng.uniform(0.3, 0.6)))
        lines.append({"text": text, "start": pos / RATE, "end": (pos + len(x)) / RATE})
        voice_parts.append((pos, x))
        pos += len(x) + int((1.6 if not seed else rng.uniform(0.9, 2.2)) * RATE)
    n = pos + RATE
    voice = np.zeros(n, np.float32)
    for p, x in voice_parts:
        voice[p:p + len(x)] += x
    bg = music(n, seed) * (1.0 if not seed else float(rng.uniform(0.7, 1.6)))
    return {"voice": voice, "bg": bg, "lines": lines, "n": n}


def render_movie(master, out, intro=5.0, speed=1.0, layout="stereo", commentary=True, work=None):
    """Write a test movie. Returns dict with paths of the true stems (aligned to the movie's audio)."""
    out = Path(out)
    work = Path(work or out.parent)
    pad = np.zeros(int(intro * RATE), np.float32)
    logo = np.sin(2 * np.pi * 440 * np.arange(len(pad)) / RATE).astype(np.float32) * 0.1 * \
        (np.arange(len(pad)) < RATE)          # 1 s "studio logo" chime, then silence
    voice = np.concatenate([pad * 0, master["voice"]])
    bg = np.concatenate([logo + pad * 0, master["bg"]])
    if speed != 1.0:                           # PAL-style speed-up (pitch goes up too)
        from scipy.signal import resample
        m = int(round(len(voice) / speed))
        voice = resample(voice, m).astype(np.float32)
        bg = resample(bg, m).astype(np.float32)
    n = len(voice)
    if layout == "stereo":
        mix = np.stack([voice * 0.9 + bg, voice * 0.9 + bg * 0.95], 1)
        vstem = np.stack([voice * 0.9, voice * 0.9], 1)
        bstem = np.stack([bg, bg * 0.95], 1)
    else:  # 5.1(side): FL FR FC LFE SL SR, dialogue in center with a little bleed to the fronts
        z = np.zeros(n, np.float32)
        vstem = np.stack([voice * 0.2, voice * 0.2, voice, z, z, z], 1)
        bstem = np.stack([bg, bg * 0.95, bg * 0.15, bg * 0.3, bg * 0.5, bg * 0.5], 1)
        mix = vstem + bstem
    stems = {"voice": work / f"{out.stem}.voice.wav", "bg": work / f"{out.stem}.bg.wav",
             "mix": work / f"{out.stem}.mix.wav"}
    sf.write(stems["voice"], vstem, RATE, subtype="FLOAT")
    sf.write(stems["bg"], bstem, RATE, subtype="FLOAT")
    sf.write(stems["mix"], mix, RATE, subtype="FLOAT")
    dur = n / RATE
    srt = work / f"{out.stem}.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:04,000\nsubtitle test\n")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=160x90:r=24:d={dur:.3f}",
           "-i", str(stems["mix"])]
    if commentary:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "sine=f=330:r=48000"]
    cmd += ["-i", str(srt), "-map", "0:v", "-map", "1:a"]
    if commentary:
        cmd += ["-map", "2:a"]
    cmd += ["-map", f"{3 if commentary else 2}:s", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35"]
    if layout == "stereo":
        cmd += ["-c:a:0", "aac", "-b:a:0", "256k"]
    else:
        cmd += ["-c:a:0", "ac3", "-b:a:0", "448k", "-channel_layout:a:0", "5.1(side)"]
    if commentary:
        cmd += ["-c:a:1", "aac", "-b:a:1", "64k", "-metadata:s:a:1", "title=Director commentary",
                "-disposition:a:1", "0"]
    cmd += ["-c:s", "srt", "-metadata:s:a:0", "language=eng", "-disposition:a:0", "default", str(out)]
    subprocess.run(cmd, check=True)
    meta = {"intro": intro, "speed": speed, "layout": layout,
            "lines": [{"text": l["text"], "start": (l["start"] + intro) / speed, "end": (l["end"] + intro) / speed}
                      for l in master["lines"]],
            "stems": {k: str(v) for k, v in stems.items()}}
    Path(str(out) + ".truth.json").write_text(json.dumps(meta, indent=1))
    return meta


def have_tools():
    return all(shutil.which(t) for t in ("ffmpeg", "espeak-ng"))
