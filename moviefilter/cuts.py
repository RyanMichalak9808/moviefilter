"""File formats.

<movie>.cuts.json   SHAREABLE. Anchor lines + time ranges relative to anchor 0.
                    Contains no cuss-word text, only levels and times.
<movie>.review.json PRIVATE. Every detection with its word/context + your edits
                    (drops/adds). Edits survive re-running analyze.
"""
from pathlib import Path

from . import __version__
from .util import merge_ranges, read_json, write_json

FORMAT = "moviefilter-cuts"
PAD_BEFORE, PAD_AFTER = 0.06, 0.10   # margin around word timestamps (snapping adds more)
FALLBACK_PAD = (0.10, 0.25)          # used when no voice stem is available


def cuts_path(movie):
    return Path(movie).with_suffix(".cuts.json")


def review_path(movie):
    return Path(movie).with_suffix(".review.json")


def load_review(movie):
    return read_json(review_path(movie), {"version": 1, "detections": [], "edits": {"drop": [], "add": []}})


def save_review(movie, review):
    write_json(review_path(movie), review)


def active_items(review, levels=None):
    """Detections that are not dropped (and match `levels`), plus manual additions."""
    drops = [d["t"] for d in review["edits"]["drop"]]
    accepts = [a["t"] for a in review["edits"].get("accept", [])]
    out = []
    for d in review["detections"]:
        if levels and d["level"] not in levels:
            continue
        if any(d["start"] - 0.05 <= t <= d["end"] + 0.05 for t in drops):
            continue
        if d.get("suggest") and not any(d["start"] - 0.05 <= t <= d["end"] + 0.05 for t in accepts):
            continue
        out.append(d)
    for a in review["edits"]["add"]:
        out.append({"start": a["start"], "end": a["end"], "level": a.get("level", "manual"),
                    "kind": "manual", "text": "", "flags": [], "prob": 1.0})
    return sorted(out, key=lambda d: d["start"])


def to_ranges(items):
    """Detections -> padded ranges (absolute, this copy's clock)."""
    rs = [{"start": max(0.0, d["start"] - PAD_BEFORE), "end": d["end"] + PAD_AFTER,
           "core": [d["start"], d["end"]], "level": d["level"],
           "src": "manual" if d.get("kind") == "manual" else "auto"} for d in items]
    return merge_ranges(rs, gap=0.15)


def build_cuts(title, language, runtime, anchors, ranges):
    """anchors: [{text, time}] absolute; ranges absolute. Stored relative to anchors[0]."""
    t0 = anchors[0]["time"]
    return {
        "format": FORMAT, "version": 2, "created_with": f"moviefilter {__version__}",
        "title": title, "language": language, "runtime": round(runtime, 3), "origin": round(t0, 3),
        "anchors": [{"text": a["text"], "t": round(a["time"] - t0, 3)} for a in anchors],
        "ranges": [{"start": round(r["start"] - t0, 3), "end": round(r["end"] - t0, 3),
                    "core": [round(r["core"][0] - t0, 3), round(r["core"][1] - t0, 3)],
                    "level": r["level"], "levels": r.get("levels", [r["level"]]), "src": r["src"]}
                   for r in ranges],
    }


def load_cuts(path):
    data = read_json(path)
    if data is None:
        raise SystemExit(f"cuts file not found: {path}")
    if data.get("format") == FORMAT:
        if data.get("version", 2) > 2:
            raise SystemExit("This cuts file was made by a newer moviefilter; please upgrade.")
        return data
    # version 0.2/0.3 single-anchor files
    if "anchor" in data:
        items = data.get("cusses") or data.get("mutes") or []
        return {
            "format": FORMAT, "version": 2, "title": data.get("title", ""), "language": "en",
            "runtime": None, "origin": None,
            "anchors": [{"text": data["anchor"]["text"], "t": 0.0}],
            "ranges": [{"start": r["start"], "end": r["end"], "core": [r["start"], r["end"]],
                        "level": "strong", "levels": ["strong"], "src": "auto"} for r in items],
        }
    raise SystemExit(f"{path} is not a moviefilter cuts file")
