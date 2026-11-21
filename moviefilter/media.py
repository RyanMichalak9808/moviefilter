"""ffmpeg/ffprobe helpers: probing, stream choice, sample-accurate decoding, encoding, muxing.

Clock convention: time 0 == the first decoded sample of the chosen audio stream.
Every audio path in this tool decodes from the start of the stream, so all analysis,
patching and verification are sample-aligned with each other.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .util import log

CLEAN_TAG = "moviefilter"          # stream tag key marking our clean track
VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts", ".m2ts", ".wmv"}

# Channel order of common FFmpeg layouts (used to find front/center/surround channels).
LAYOUTS = {
    "mono": ["FC"], "stereo": ["FL", "FR"], "2.1": ["FL", "FR", "LFE"], "3.0": ["FL", "FR", "FC"],
    "quad": ["FL", "FR", "BL", "BR"], "4.0": ["FL", "FR", "FC", "BC"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"], "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"], "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
}
DEFAULT_LAYOUT = {1: "mono", 2: "stereo", 3: "2.1", 4: "quad", 5: "5.0(side)", 6: "5.1(side)", 7: "6.1", 8: "7.1"}


def require_tools():
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} - install FFmpeg and make sure it is on PATH.")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", str(path)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def audio_streams(info):
    """Audio streams in order; 'order' is the N in ffmpeg's 0:a:N."""
    res = []
    for s in info["streams"]:
        if s.get("codec_type") != "audio":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        title = tags.get("title") or ("" if tags.get("handler_name", "").endswith("Handler")
                                      else tags.get("handler_name", ""))       # MP4 keeps names here
        res.append({
            "order": len(res), "index": s["index"], "codec": s.get("codec_name"),
            "channels": int(s.get("channels") or 2), "rate": int(s.get("sample_rate") or 48000),
            "layout": s.get("channel_layout") or "", "language": tags.get("language", "und"),
            "title": title, "default": bool((s.get("disposition") or {}).get("default")),
            "start": float(s.get("start_time") or 0.0),
            "duration": float(s.get("duration") or info.get("format", {}).get("duration") or 0),
            "clean": tags.get(CLEAN_TAG) == "clean" or "moviefilter" in title.lower()
                     or title.lower().startswith("clean (no profanity)"),
        })
    return res


def choose_stream(streams, order=None, language=None):
    """Pick the source track: explicit order > default track > language match > first.

    Our own clean tracks and commentary tracks are never picked automatically.
    """
    if not streams:
        raise SystemExit("No audio stream found.")
    if order is not None:
        if not 0 <= order < len(streams):
            raise SystemExit(f"--audio-track {order} out of range (file has {len(streams)} audio tracks)")
        return streams[order]
    pool = [s for s in streams if not s["clean"] and "commentary" not in s["title"].lower()] or streams
    if language:
        lang_pool = [s for s in pool if s["language"].startswith(language[:2])]
        pool = lang_pool or pool
    for s in pool:
        if s["default"]:
            return s
    return pool[0]


def channel_names(stream):
    lay = stream["layout"].split("(")[0] if stream["layout"] not in LAYOUTS else stream["layout"]
    names = LAYOUTS.get(stream["layout"]) or LAYOUTS.get(lay)
    if not names or len(names) != stream["channels"]:
        names = LAYOUTS[DEFAULT_LAYOUT.get(stream["channels"], "stereo")]
        if len(names) != stream["channels"]:
            names = [f"C{i}" for i in range(stream["channels"])]
    return names


def channel_groups(stream, surrounds=False):
    """Channel groups to separate: front pair, center (dialogue), optionally surrounds. LFE never."""
    names = channel_names(stream)
    idx = {n: i for i, n in enumerate(names)}
    if stream["channels"] == 1:
        return [(0,)]
    groups = []
    if "FL" in idx and "FR" in idx:
        groups.append((idx["FL"], idx["FR"]))
    if "FC" in idx:
        groups.append((idx["FC"],))
    if surrounds:
        for a, b in (("SL", "SR"), ("BL", "BR"), ("FLC", "FRC")):
            if a in idx and b in idx:
                groups.append((idx[a], idx[b]))
    if not groups:          # unknown layout: treat channels pairwise
        groups = [tuple(range(i, min(i + 2, stream["channels"]))) for i in range(0, stream["channels"], 2)]
    return sorted(groups)


def dialogue_channels(stream):
    names = channel_names(stream)
    if "FC" in names and len(names) > 2:
        return [names.index("FC")]
    return list(range(min(2, len(names))))


# --------------------------------------------------------------------------- decoding
def decode_cmd(path, order, rate, channels, sample_fmt="f32le"):
    codec = {"f32le": "pcm_f32le", "s16le": "pcm_s16le"}[sample_fmt]
    return ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{order}",
            "-vn", "-sn", "-dn", "-ac", str(channels), "-ar", str(rate),
            "-f", sample_fmt, "-acodec", codec, "-"]


def iter_blocks(path, stream, block_sec=2.0):
    """Yield (start_sample, float32 array [n, channels]) for the whole stream at native rate."""
    ch, rate = stream["channels"], stream["rate"]
    proc = subprocess.Popen(decode_cmd(path, stream["order"], rate, ch), stdout=subprocess.PIPE)
    frame_bytes = 4 * ch
    want = int(block_sec * rate) * frame_bytes
    pos, buf = 0, b""
    finished = False
    try:
        while True:
            chunk = proc.stdout.read(want - len(buf))
            if not chunk:
                break
            buf += chunk
            if len(buf) < want:
                continue
            arr = np.frombuffer(buf, dtype=np.float32).reshape(-1, ch)
            yield pos, arr
            pos += len(arr)
            buf = b""
        usable = len(buf) - len(buf) % frame_bytes
        if usable:
            arr = np.frombuffer(buf[:usable], dtype=np.float32).reshape(-1, ch)
            yield pos, arr
        finished = True
    finally:
        if not finished:          # caller stopped early: stop ffmpeg quietly (no broken-pipe spam)
            proc.kill()
        proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg decode failed ({rc}) for {path}")


def decode_16k(path, stream, out_file):
    """Decode the stream once to 16 kHz mono int16 (used for ASR, anchors and review clips)."""
    out_file = Path(out_file)
    tmp = out_file.with_name(out_file.name + ".part")
    cmd = decode_cmd(path, stream["order"], 16000, 1, "s16le")
    with open(tmp, "wb") as f:
        subprocess.run(cmd, stdout=f, check=True)
    tmp.replace(out_file)


class Audio16k:
    """Random access into the cached 16 kHz mono decode."""
    RATE = 16000

    def __init__(self, path):
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=np.int16, mode="r")

    @property
    def duration(self):
        return len(self.data) / self.RATE

    def read(self, start, end):
        a = max(0, int(start * self.RATE))
        b = min(len(self.data), int(end * self.RATE))
        return np.asarray(self.data[a:b], dtype=np.float32) / 32768.0


# --------------------------------------------------------------------------- encoding
def default_codec(channels):
    return ("aac", "256k") if channels <= 2 else ("ac3", "640k")


def encoder(out_file, rate, channels, layout, codec=None, bitrate=None):
    """Start an ffmpeg process that reads float32 PCM on stdin and writes an encoded .mka."""
    c, br = default_codec(channels)
    codec = codec or c
    bitrate = bitrate or br
    lay = layout if layout in LAYOUTS else DEFAULT_LAYOUT.get(channels)
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ch_layout", lay,
           "-i", "pipe:0", "-c:a", codec]
    if codec not in ("flac", "pcm_s16le", "pcm_s24le"):
        cmd += ["-b:a", bitrate]
    if codec == "flac":
        cmd += ["-sample_fmt", "s32", "-compression_level", "5"]
    cmd += [str(out_file)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def best_lag(ref, x, max_lag=8192):
    """Delay of x relative to ref in samples (FFT cross-correlation)."""
    n = len(ref) + len(x)
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lags = np.concatenate([c[:max_lag + 1], c[-max_lag:]])
    idx = int(np.argmax(lags))
    return idx if idx <= max_lag else idx - len(lags)


_DELAY_CACHE = {}


def codec_delay(codec, bitrate, rate, channels, layout, tmpdir):
    """Samples of priming the encoder adds (AAC ~1024, AC-3 ~256, FLAC 0), measured once."""
    key = (codec, bitrate, rate, channels)
    if key in _DELAY_CACHE:
        return _DELAY_CACHE[key]
    rng = np.random.default_rng(1)
    sig = np.zeros((rate * 2, channels), np.float32)
    burst = rng.standard_normal((rate // 2, channels)).astype(np.float32) * 0.3
    sig[rate // 4: rate // 4 + len(burst)] = burst
    probe_file = Path(tmpdir) / f"delay_{codec}.mka"
    enc = encoder(probe_file, rate, channels, layout, codec, bitrate)
    enc.stdin.write(sig.tobytes())
    enc.stdin.close()
    enc.wait()
    raw = subprocess.run(decode_cmd(probe_file, 0, rate, channels), capture_output=True, check=True).stdout
    dec = np.frombuffer(raw, np.float32).reshape(-1, channels)
    lag = best_lag(sig.mean(1), dec.mean(1)[: len(sig) + 8192])
    probe_file.unlink(missing_ok=True)
    _DELAY_CACHE[key] = max(0, lag)
    return _DELAY_CACHE[key]


def mux(movie, clean_audio, output, stream, info, title, replace_audio=False, make_default=False,
        drop_existing_clean=True):
    """Copy every original stream, add (or swap in) the clean track."""
    movie, output = Path(movie), Path(output)
    streams = audio_streams(info)
    drop = [s["index"] for s in streams if s["clean"]] if drop_existing_clean else []
    mp4ish = output.suffix.lower() in (".mp4", ".m4v", ".mov")
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(movie)]
    if stream["start"] > 0:
        cmd += ["-itsoffset", f"{stream['start']:.6f}"]
    cmd += ["-i", str(clean_audio)]
    if mp4ish:
        cmd += ["-map", "0:v?", "-map", "0:s?"]
        if not replace_audio:
            cmd += ["-map", "0:a?"]
    else:
        cmd += ["-map", "0"]
        if replace_audio:
            cmd += ["-map", "-0:a"]
    for idx in drop:
        if not replace_audio:
            cmd += ["-map", f"-0:{idx}"]
    cmd += ["-map", "1:a:0", "-c", "copy"]
    kept_audio = 0 if replace_audio else len([s for s in streams if s["index"] not in drop])
    n = kept_audio
    cmd += [f"-metadata:s:a:{n}", f"title={title}", f"-metadata:s:a:{n}", f"handler_name={title}",
            f"-metadata:s:a:{n}", f"language={stream['language']}",
            f"-metadata:s:a:{n}", f"{CLEAN_TAG}=clean"]
    if make_default or replace_audio:
        for i in range(n):
            cmd += [f"-disposition:a:{i}", "0"]
        cmd += [f"-disposition:a:{n}", "default"]
    else:
        cmd += [f"-disposition:a:{n}", "0"]
    cmd += [str(output)]
    subprocess.run(cmd, check=True)


def check_space(path, need_bytes, label):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < need_bytes * 1.1:
        raise SystemExit(f"Not enough disk space for {label} in {path}: need ~{need_bytes / 1e9:.1f} GB, "
                         f"have {free / 1e9:.1f} GB. Use --workdir/--output elsewhere, or --no-space-check.")
    log(f"  disk ok for {label}: need ~{need_bytes / 1e9:.2f} GB, free {free / 1e9:.1f} GB")
