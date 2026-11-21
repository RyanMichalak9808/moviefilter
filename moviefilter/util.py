"""Small shared helpers: time formatting, logging, progress, JSON, range merging."""
import json
import os
import sys
import time
from pathlib import Path

QUIET = False


def log(msg=""):
    if not QUIET:
        print(msg, file=sys.stderr, flush=True)


def parse_time(s):
    """'83.5' | '1:23' | '1:02:10.5' -> seconds (float)."""
    s = str(s).strip()
    if not s:
        raise ValueError("empty time")
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def parse_range(s):
    """'1:02:03.5-1:02:04.2' -> (start, end)."""
    if "-" not in s.lstrip("-"):
        raise ValueError(f"expected START-END, got {s!r}")
    a, b = s.rsplit("-", 1)
    a, b = parse_time(a), parse_time(b)
    if b <= a:
        raise ValueError(f"range end must be after start: {s!r}")
    return a, b


def fmt_time(t):
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def write_json(path, data):
    """Atomic JSON write (never leaves a half-written file behind)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def merge_ranges(ranges, gap=0.0):
    """Merge dicts with start/end that overlap or sit within `gap` seconds.

    Keeps the first range's extra keys; collects 'level' values into 'levels'.
    """
    out = []
    for r in sorted(ranges, key=lambda r: (r["start"], r["end"])):
        if out and r["start"] <= out[-1]["end"] + gap:
            last = out[-1]
            last["end"] = max(last["end"], r["end"])
            if "core" in r and "core" in last:
                last["core"] = [min(last["core"][0], r["core"][0]), max(last["core"][1], r["core"][1])]
            if r.get("level") and r["level"] not in last.setdefault("levels", [last.get("level")]):
                last["levels"].append(r["level"])
        else:
            out.append(dict(r))
    return out


class Progress:
    """Minimal progress line with ETA (no extra dependency)."""

    def __init__(self, label, total, unit=""):
        self.label, self.total, self.unit = label, max(total, 1e-9), unit
        self.n = 0.0
        self.t0 = time.time()
        self._last = 0.0
        self.tty = sys.stderr.isatty()

    def update(self, n=1, force=False):
        self.n += n
        now = time.time()
        if QUIET or (not force and now - self._last < (0.5 if self.tty else 15)):
            return
        self._last = now
        frac = min(self.n / self.total, 1.0)
        el = now - self.t0
        eta = el / frac - el if frac > 0.001 else 0
        line = f"  {self.label}: {frac * 100:5.1f}%  elapsed {int(el)}s  eta {int(eta)}s"
        print(("\r" + line) if self.tty else line, end="" if self.tty else "\n",
              file=sys.stderr, flush=True)

    def done(self):
        self.n = self.total
        self.update(0, force=True)
        if self.tty and not QUIET:
            print(file=sys.stderr)
