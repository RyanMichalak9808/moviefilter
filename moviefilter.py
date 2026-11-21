#!/usr/bin/env python3
"""
moviefilter.py - remove cuss words from a movie you own WITHOUT leaving a silent gap.

Instead of muting, the voice is separated from the music/effects (Demucs) around each
cuss word, and only the voice is dropped.  The background keeps playing underneath.

  analyze : extract audio -> split into 10-min tracks -> Whisper word timestamps
            -> list of cuss-word time ranges -> <movie>.cuts.json
            (times are stored relative to an ANCHOR line so the file works on other
             rips of the same cut of the film and is safe to share)
  apply   : find the anchor in your copy, separate vocals around each cuss word,
            swap in the vocal-free background (short crossfade), mux back with the
            original video copied untouched.

Requirements
  ffmpeg on PATH
  pip install faster-whisper demucs soundfile numpy
  (demucs pulls in PyTorch; CPU works, GPU is much faster)

Usage
  python moviefilter.py analyze Movie.mkv [--model small.en] [--mild] [--anchor-text "..."]
  python moviefilter.py apply   Movie.mkv Movie.cuts.json -o Movie.clean.mkv

Notes
  * Output audio is the first audio track downmixed to stereo 44.1 kHz.
  * Only windows around cuss words go through Demucs, so it's far faster than
    separating the whole film.
  * Anchor transfer only works between copies of the same edit (not theatrical vs extended).
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHUNK_SECONDS = 600                  # audio "tracks" are 10-minute pieces
PAD_BEFORE, PAD_AFTER = 0.10, 0.25   # padding around each cussed word
CONTEXT = 4.0                        # seconds of context given to Demucs each side
FADE = 0.015                         # crossfade length (s) at splice points

# Edit freely. Matched on lower-cased words, punctuation stripped.
STRONG = [r"fuck\w*", r"motherfuck\w*", r"shit\w*", r"bullshit", r"bitch\w*",
          r"asshole\w*", r"dumbass", r"jackass", r"bastard\w*", r"cunt\w*",
          r"cock(sucker)?\w*", r"dick(head)?s?", r"pussy", r"whore\w*", r"slut\w*",
          r"goddamn\w*", r"piss\w*", r"wank\w*", r"twat"]
MILD = [r"damn\w*", r"hell", r"ass", r"crap", r"bloody", r"jesus", r"christ"]


# --------------------------------------------------------------------------- utils
def run(cmd):
    subprocess.run(cmd, check=True)


def parse_time(s):
    sec = 0.0
    for p in str(s).split(":"):
        sec = sec * 60 + float(p)
    return sec


def fmt_time(t):
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def merge_ranges(ranges, gap=0.0):
    out = []
    for r in sorted(ranges, key=lambda r: r["start"]):
        if out and r["start"] <= out[-1]["end"] + gap:
            out[-1]["end"] = max(out[-1]["end"], r["end"])
        else:
            out.append(dict(r))
    return out


def rebase(ranges, t0, to_relative=True):
    d = -t0 if to_relative else t0
    return [{"start": r["start"] + d, "end": r["end"] + d} for r in ranges]


# ----------------------------------------------------------- audio split + transcribe
def split_audio(video, outdir, start=0.0, duration=None):
    """First audio stream -> 16 kHz mono WAV tracks (for Whisper)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "segment",
            "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
            str(Path(outdir) / "chunk_%04d.wav")]
    run(cmd)
    return sorted(Path(outdir).glob("chunk_*.wav"))


def transcribe(video, model_name="small.en", start=0.0, duration=None):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, compute_type="int8")
    words = []
    with tempfile.TemporaryDirectory() as td:
        chunks = split_audio(video, td, start, duration)
        for i, wav in enumerate(chunks):
            base = start + i * CHUNK_SECONDS
            print(f"  transcribing track {i + 1}/{len(chunks)} ...", file=sys.stderr)
            segs, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=True,
                                       language="en")
            for s in segs:
                for w in s.words or []:
                    words.append({"start": base + w.start, "end": base + w.end,
                                  "word": w.word.strip()})
    return words


# ------------------------------------------------------------------ profanity + anchor
def find_cusses(words, mild=False):
    pat = re.compile(r"^(?:%s)$" % "|".join(STRONG + (MILD if mild else [])))
    return [{"start": max(0.0, w["start"] - PAD_BEFORE), "end": w["end"] + PAD_AFTER}
            for w in words if pat.match(norm(w["word"]))]


def locate_anchor(words, text):
    target = " ".join(norm(t) for t in text.split() if norm(t))
    n = len(target.split())
    stream = [norm(w["word"]) for w in words]
    best, best_i = 0.0, None
    for i in range(max(1, len(stream) - n + 1)):
        score = difflib.SequenceMatcher(None, target, " ".join(stream[i:i + n])).ratio()
        if score > best + 1e-9:
            best, best_i = score, i
        if best > 0.98:
            break
    if best_i is None or best < 0.7:
        raise SystemExit(f"Anchor {text!r} not found (best {best:.2f}). Use --anchor-time.")
    print(f"  anchor at {fmt_time(words[best_i]['start'])} (score {best:.2f})", file=sys.stderr)
    return words[best_i]["start"]


# ------------------------------------------------------- vocal-removal splice (core)
def crossfade_replace(orig, rep, fade_n, fade_in=True, fade_out=True):
    """Blend `rep` over `orig` (both float arrays [n, ch]) with linear edge crossfades."""
    import numpy as np
    n = min(len(orig), len(rep))
    orig, rep = orig[:n], rep[:n]
    w = np.ones(n, dtype=np.float32)
    f = min(fade_n, n // 2)
    if f > 0 and fade_in:
        w[:f] = np.linspace(0, 1, f, dtype=np.float32)
    if f > 0 and fade_out:
        w[n - f:] = np.linspace(1, 0, f, dtype=np.float32)
    return orig * (1 - w)[:, None] + rep * w[:, None]


def plan_windows(ranges, total_sec):
    """Group cuss ranges into Demucs windows (range +/- CONTEXT, overlapping ones merged)."""
    wins = []
    for r in sorted(ranges, key=lambda r: r["start"]):
        s, e = max(0.0, r["start"] - CONTEXT), min(total_sec, r["end"] + CONTEXT)
        if wins and s <= wins[-1]["end"]:
            wins[-1]["end"] = max(wins[-1]["end"], e)
            wins[-1]["ranges"].append(r)
        else:
            wins.append({"start": s, "end": e, "ranges": [r]})
    return wins


def remove_vocals_at(wav_path, ranges, workdir, model="htdemucs"):
    """Edit wav_path in place: inside each range, vocals are replaced by Demucs' background."""
    import soundfile as sf
    info = sf.info(wav_path)
    sr, total = info.samplerate, info.frames
    wins = plan_windows(ranges, total / sr)
    win_dir, out_dir = Path(workdir) / "win", Path(workdir) / "sep"
    win_dir.mkdir(exist_ok=True)

    with sf.SoundFile(wav_path) as f:
        for i, w in enumerate(wins):
            f.seek(int(w["start"] * sr))
            data = f.read(int((w["end"] - w["start"]) * sr), dtype="float32", always_2d=True)
            sf.write(win_dir / f"w{i:05d}.wav", data, sr)
    print(f"  separating vocals in {len(wins)} windows ...", file=sys.stderr)
    run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", model,
         "-o", str(out_dir), *sorted(map(str, win_dir.glob("w*.wav")))])

    fade_n = int(FADE * sr)
    with sf.SoundFile(wav_path, "r+") as f:
        for i, w in enumerate(wins):
            bg, bg_sr = sf.read(out_dir / model / f"w{i:05d}" / "no_vocals.wav",
                                dtype="float32", always_2d=True)
            assert bg_sr == sr, "sample-rate mismatch"
            w0 = int(w["start"] * sr)
            for r in w["ranges"]:
                a, b = int(r["start"] * sr), min(int(r["end"] * sr), total)
                a0, b0 = max(a - fade_n, 0), min(b + fade_n, total)
                f.seek(a0)
                orig = f.read(b0 - a0, dtype="float32", always_2d=True)
                rep = bg[a0 - w0: b0 - w0]
                mixed = crossfade_replace(orig, rep, fade_n)
                f.seek(a0)
                f.write(mixed)


# ------------------------------------------------------------------------ commands
def cmd_analyze(a):
    movie = Path(a.movie)
    words = transcribe(movie, a.model)
    movie.with_suffix(".transcript.json").write_text(json.dumps(words))
    if a.anchor_text:
        text = a.anchor_text
        t0 = parse_time(a.anchor_time) if a.anchor_time else locate_anchor(words, text)
    else:
        first = words[:8]
        if len(first) < 8:
            raise SystemExit("Not enough speech detected; pass --anchor-text.")
        text, t0 = " ".join(w["word"] for w in first), first[0]["start"]
    cusses = merge_ranges(find_cusses(words, a.mild), gap=0.3)
    cuts = {"title": movie.stem,
            "anchor": {"text": text, "note": "times are seconds after this line starts"},
            "cusses": rebase(cusses, t0)}
    out = movie.with_suffix(".cuts.json")
    out.write_text(json.dumps(cuts, indent=2))
    print(f"Anchor {text!r} @ {fmt_time(t0)}\n{len(cusses)} cuss ranges -> {out}")


def count_audio_streams(movie):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "csv=p=0", str(movie)],
                         check=True, capture_output=True, text=True).stdout
    return len([l for l in out.splitlines() if l.strip()])


def cmd_apply(a):
    cuts = json.loads(Path(a.cuts).read_text())
    if a.anchor_time:
        t0 = parse_time(a.anchor_time)
    else:
        cache = Path(a.movie).with_suffix(".transcript.json")
        words = (json.loads(cache.read_text()) if cache.exists()
                 else transcribe(a.movie, "small.en", duration=900))
        t0 = locate_anchor(words, cuts["anchor"]["text"])
    ranges = merge_ranges(rebase(cuts["cusses"], t0, to_relative=False), gap=0.3)
    ranges = [r for r in ranges if r["end"] > 0]

    with tempfile.TemporaryDirectory(dir=a.tmp) as td:
        wav = Path(td) / "audio.wav"
        print("Extracting audio ...", file=sys.stderr)
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a.movie), "-map", "0:a:0",
             "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(wav)])
        if ranges:
            remove_vocals_at(wav, ranges, td, a.demucs_model)
        print("Muxing ...", file=sys.stderr)
        if a.replace_audio:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a.movie), "-i", str(wav),
                 "-map", "0:v", "-map", "1:a", "-map", "0:s?",
                 "-c:v", "copy", "-c:s", "copy", "-c:a", "aac", "-b:a", "256k", str(a.output)])
        else:
            # Keep every original stream and ADD the clean mix as one more audio track.
            # Plex lists it by title in the player's audio menu.
            n = count_audio_streams(a.movie)
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(a.movie), "-i", str(wav),
                 "-map", "0", "-map", "1:a:0", "-c", "copy",
                 f"-c:a:{n}", "aac", f"-b:a:{n}", "256k",
                 f"-metadata:s:a:{n}", f"title={a.track_title}",
                 f"-metadata:s:a:{n}", "language=eng",
                 f"-disposition:a:{n}", "0",   # original stays the default track
                 str(a.output)])
    print(f"Done: {a.output} ({len(ranges)} cuss ranges cleaned)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("analyze")
    s.add_argument("movie")
    s.add_argument("--model", default="small.en", help="faster-whisper model")
    s.add_argument("--mild", action="store_true", help="also catch damn/hell/etc.")
    s.add_argument("--anchor-text", help="line to anchor on (default: first 8 spoken words)")
    s.add_argument("--anchor-time", help="clock time of that line (skips searching)")
    s.set_defaults(f=cmd_analyze)

    s = sp.add_parser("apply")
    s.add_argument("movie")
    s.add_argument("cuts")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--anchor-time", help="clock time of the anchor line in this copy")
    s.add_argument("--demucs-model", default="htdemucs")
    s.add_argument("--tmp", help="scratch dir (needs ~1.3 GB per 2 h of film)")
    s.add_argument("--track-title", default="Clean (no profanity)",
                   help="name Plex shows in the audio menu")
    s.add_argument("--replace-audio", action="store_true",
                   help="swap the audio instead of adding a second track")
    s.set_defaults(f=cmd_apply)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
