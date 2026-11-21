"""Real Whisper + Demucs on synthetic movies with known ground truth.

Run:  pytest --slow tests/test_integration.py        (MF_SEED=n for a different rendering)
Takes ~10-15 min on a 4-core CPU; downloads ~600 MB of models the first time.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

fixtures = pytest.importorskip("fixtures")
if not fixtures.have_tools():
    pytest.skip("needs ffmpeg + espeak-ng", allow_module_level=True)

import evaluate  # noqa: E402

from moviefilter import pipeline as pl  # noqa: E402
from moviefilter.cuts import cuts_path  # noqa: E402
from moviefilter.wordlist import Matcher  # noqa: E402

SEED = int(os.environ.get("MF_SEED", "0"))
STRONG = Matcher(levels=("strong", "slurs"))


def strong_lines(meta):
    """Script lines that contain a strong cuss word (ground truth)."""
    out = []
    for ln in meta["lines"]:
        ws = [{"word": w, "start": 0, "end": 0} for w in ln["text"].split()]
        if STRONG.find(ws):
            out.append(ln)
    return out


@pytest.fixture(scope="module")
def movies(tmp_path_factory):
    d = tmp_path_factory.mktemp(f"seed{SEED}")
    os.environ["MOVIEFILTER_CACHE"] = str(d / "cache")
    master = fixtures.build_master(d / "work", seed=SEED)
    variants = {"A": dict(intro=5.0), "B": dict(intro=17.3, commentary=False),
                "C": dict(intro=5.0, speed=25 / 23.976), "D": dict(intro=5.0, layout="5.1")}
    return {k: (d / f"{k}.mkv", fixtures.render_movie(master, d / f"{k}.mkv", **v)) for k, v in variants.items()}


@pytest.fixture(scope="module")
def eng():
    return pl.Engines()


def check_quality(movie, out, rep, meta):
    s = evaluate.score(movie, out, rep)
    print(json.dumps({k: v for k, v in s.items() if k != "ranges"}),
          [(r["start"], r["status"], r["voice_leak_db"], r["bg_kept_db"]) for r in s["ranges"]])
    surround = meta["layout"] != "stereo"
    assert s["sync_error_samples"] == 0
    # rest of the track = original. Lossy re-encoding alone measures -15..-35 dB on these pure
    # synthetic tones (AC-3 round trip with no edits: ~-20 dB); misalignment would be ~0 dB.
    assert s["outside_diff_db"] < -15
    for r in s["ranges"]:
        assert r["voice_leak_db"] < -20, r                    # the original voice is gone
        assert -4.0 < r["bg_kept_db"] < 3.0, r                # background still there (a mute would be < -20)
    assert sum(r["bg_corr"] for r in s["ranges"]) / len(s["ranges"]) > (0.75 if surround else 0.9)
    # every scripted cuss line got cleaned
    for ln in strong_lines(meta):
        assert any(r["start"] < ln["end"] and r["end"] > ln["start"] for r in rep["ranges"]), ln["text"]
    # a fresh listen hears no strong cussing, and most normal words survive
    words, dets = evaluate.transcribe_clean(out)
    # only count words heard where someone is actually talking: Whisper (vad off) sometimes
    # "hears" a word in the intro chime/silence, which is untouched original audio anyway
    in_speech = [d for d in dets if any(d["start"] < ln["end"] and d["end"] > ln["start"] for ln in meta["lines"])]
    assert in_speech == [], [d["text"] for d in in_speech]
    total = sum(len(ln["text"].split()) for ln in meta["lines"])
    assert len(words) > 0.85 * total - 10


def test_analyze_finds_every_cuss_line(movies, eng):
    movie, meta = movies["A"]
    review = pl.analyze(movie, eng)
    strong = [d for d in review["detections"] if d["level"] in ("strong", "slurs") and not d.get("suggest")]
    for ln in strong_lines(meta):
        assert any(d["start"] < ln["end"] and d["end"] > ln["start"] for d in strong), f"missed: {ln['text']}"
    clean = [ln for ln in meta["lines"] if ln not in strong_lines(meta)]
    for d in strong:
        assert not any(d["start"] < ln["end"] and d["end"] > ln["start"] for ln in clean), f"false hit: {d}"
    assert len(review["anchors"]) >= 2


def test_apply_same_copy(movies, eng):
    movie, meta = movies["A"]
    out = movie.with_name("A.clean.mkv")
    rep = pl.apply(movie, cuts_path(movie), eng, output=out)
    assert rep["summary"]["ranges"] >= len(strong_lines(meta))
    check_quality(movie, out, rep, meta)


@pytest.mark.parametrize("variant", ["B", "C", "D"])
def test_shared_cuts_on_other_copies(movies, eng, variant):
    src, _ = movies["A"]
    shared = src.with_name("shared.cuts.json")
    shared.write_text(cuts_path(src).read_text())
    movie, meta = movies[variant]
    out = movie.with_name(f"{variant}.clean.mkv")
    rep = pl.apply(movie, shared, eng, output=out)
    assert all(a["found"] is not None for a in rep["anchors"]["anchors"])
    assert not [w for w in rep["anchors"]["warnings"] if "different cut" in w or "longer after" in w]
    check_quality(movie, out, rep, meta)
