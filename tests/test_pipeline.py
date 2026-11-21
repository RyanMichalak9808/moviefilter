"""End-to-end with real ffmpeg but fake ASR/separation (fast, exact, no model downloads)."""
import json

import numpy as np
import pytest
from conftest import RATE, SCRIPT, decode, fake_engines, tone_audio, write_movie

from moviefilter import media, pipeline as pl
from moviefilter.cuts import cuts_path, load_cuts, load_review, review_path

DUR = 30.0


def band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    return spec[(f > lo) & (f < hi)].sum()


@pytest.fixture
def movie(tmp_path, cache):
    audio, _, _ = tone_audio(SCRIPT, DUR)
    return write_movie(tmp_path / "Film (2020).mkv", audio)


def test_analyze_review_apply_roundtrip(movie, tmp_path):
    eng = fake_engines(SCRIPT)
    review = pl.analyze(movie, eng, stem_mode="off")
    texts = [d["text"] for d in review["detections"]]
    assert texts == ["fuck", "shit", "damn", "bitch"]
    cuts = load_cuts(cuts_path(movie))
    assert len(cuts["ranges"]) == 3                       # damn is mild -> off by default
    assert cuts["anchors"][0]["text"].startswith("the old harbor")
    assert "fuck" not in json.dumps(cuts)                 # shareable file has no cuss words in it

    # review: drop "bitch" (row 4), add a manual range, then restore
    rows = pl.review_cmd(movie, drop=[4], add=["0:00:15.0-0:00:15.5"], show=False)
    assert [r["dropped"] for r in rows if r["row_kind"] == "det"] == [False, False, False, True]
    assert len(load_cuts(cuts_path(movie))["ranges"]) == 3          # fuck, shit, manual
    # edits survive re-analysis
    pl.analyze(movie, eng, stem_mode="off")
    assert len(load_review(movie)["edits"]["drop"]) == 1
    assert len(load_cuts(cuts_path(movie))["ranges"]) == 3

    out = tmp_path / "out.mkv"
    rep = pl.apply(movie, cuts_path(movie), eng, output=out, verify="full", codec="flac")
    assert rep["summary"]["ranges"] == 3 and rep["summary"].get("clean") == 3
    orig = decode(movie, 0, 2)
    info = media.probe(out)
    streams = media.audio_streams(info)
    assert [s["clean"] for s in streams] == [False, True]
    assert streams[1]["title"] == "Clean (no profanity)" and not streams[1]["default"]
    clean = decode(out, 1, 2)
    assert len(clean) == len(orig)
    # cuss words: 3 kHz voice gone, 200 Hz background intact
    for a, b in [(10.0, 10.4), (20.0, 20.4)]:
        seg = slice(int(a * RATE), int(b * RATE))
        assert band_energy(clean[seg, 0], 2500, 3500) < 0.01 * band_energy(orig[seg, 0], 2500, 3500)
        assert band_energy(clean[seg, 0], 150, 250) == pytest.approx(band_energy(orig[seg, 0], 150, 250), rel=0.03)
    # dropped word and the neighbouring word are untouched
    for a, b in [(27.0, 27.4), (10.5, 10.85), (2.0, 6.0)]:
        seg = slice(int(a * RATE), int(b * RATE))
        assert np.allclose(clean[seg], orig[seg], atol=1e-4)
    # everything outside the patched spots is bit-identical (sample accurate, no drift)
    mask = np.ones(len(orig), bool)
    for r in rep["ranges"]:
        mask[int(r["start"] * RATE) - 2000: int(r["end"] * RATE) + 2000] = False
    assert np.abs(clean[mask] - orig[mask]).max() < 1e-4


def test_apply_is_resumable_and_cached(movie, tmp_path):
    eng = fake_engines(SCRIPT)
    pl.analyze(movie, eng, stem_mode="off")
    pl.apply(movie, cuts_path(movie), eng, output=tmp_path / "a.mkv", verify="fast", codec="flac")
    calls = eng.sep.calls
    pl.apply(movie, cuts_path(movie), eng, output=tmp_path / "b.mkv", verify="fast", codec="flac")
    assert eng.sep.calls == calls                          # second run reused every cleaned window
    assert np.array_equal(decode(tmp_path / "a.mkv", 1, 2), decode(tmp_path / "b.mkv", 1, 2))


LONG = SCRIPT + [
    {"word": w, "start": s0 + i * 0.5, "end": s0 + i * 0.5 + 0.35, "prob": 0.95}
    for s0, line in ((75.0, "captain reynolds wants the cargo moved before sunset tonight"),
                     (140.0, "remember to water the tomato plants in the garden"))
    for i, w in enumerate(line.split())
] + [{"word": "fuck", "start": 150.0, "end": 150.4, "prob": 0.9}]


@pytest.mark.parametrize("scale,shift", [(1.0, 7.3), (23.976 / 25, 7.3)], ids=["longer-intro", "pal+intro"])
def test_shared_cuts_on_other_copy(tmp_path, cache, scale, shift):
    """Same film, longer intro (and PAL speed-up): the shared file must still hit the words."""
    dur = 160.0
    audio, _, _ = tone_audio(LONG, dur)
    movie = write_movie(tmp_path / "orig.mkv", audio)
    eng = fake_engines(LONG)
    rv = pl.analyze(movie, eng, stem_mode="off")
    assert len(rv["anchors"]) == 3
    shared = tmp_path / "shared.cuts.json"
    shared.write_text(cuts_path(movie).read_text())

    audio, _, _ = tone_audio(LONG, dur * scale + shift, shift=shift, scale=scale)
    other = write_movie(tmp_path / "copy.mkv", audio)
    eng2 = fake_engines(LONG, shift=shift, scale=scale)
    out = tmp_path / "copy.clean.mkv"
    rep = pl.apply(other, shared, eng2, output=out, verify="fast", codec="flac")
    assert rep["anchors"]["anchors"][0]["found"] == pytest.approx(2.0 * scale + shift, abs=0.02)
    assert all(a["found"] is not None for a in rep["anchors"]["anchors"])
    if scale != 1.0:
        assert any("PAL" in w for w in rep["anchors"]["warnings"])
    clean = decode(out, 1, 2)
    orig = decode(other, 0, 2)
    for a, b in [(10.0, 10.4), (20.0, 20.4), (150.0, 150.4)]:
        a, b = a * scale + shift, b * scale + shift
        seg = slice(int(a * RATE), int(b * RATE))
        assert band_energy(clean[seg, 0], 2500, 3500) < 0.02 * band_energy(orig[seg, 0], 2500, 3500)


def test_surround_and_in_place_with_backup(tmp_path, cache):
    audio, _, _ = tone_audio(SCRIPT, DUR, channels=6)
    movie = write_movie(tmp_path / "surround.mkv", audio, layout="5.1(side)")
    eng = fake_engines(SCRIPT)
    pl.analyze(movie, eng, stem_mode="off")
    rep = pl.apply(movie, cuts_path(movie), eng, in_place=True, verify="fast", codec="flac", make_default=True)
    assert (tmp_path / ".moviefilter-backup" / "surround.mkv").exists()
    streams = media.audio_streams(media.probe(movie))
    assert len(streams) == 2 and streams[1]["clean"] and streams[1]["channels"] == 6
    assert streams[1]["default"] and not streams[0]["default"]
    clean, orig = decode(movie, 1, 6), decode(movie, 0, 6)
    seg = slice(int(10.0 * RATE), int(10.4 * RATE))
    assert band_energy(clean[seg, 2], 2500, 3500) < 0.01 * band_energy(orig[seg, 2], 2500, 3500)
    assert np.allclose(clean[:, 3], orig[:, 3], atol=1e-4)     # LFE untouched
    # running again replaces the clean track instead of stacking a second one
    pl.apply(movie, cuts_path(movie), eng, in_place=True, backup=False, verify="fast", codec="flac")
    assert len(media.audio_streams(media.probe(movie))) == 2
    assert rep["summary"]["ranges"] == 3


def test_lossy_output_stays_in_sync(movie, tmp_path):
    eng = fake_engines(SCRIPT)
    pl.analyze(movie, eng, stem_mode="off")
    for codec in ("aac", "ac3"):
        out = tmp_path / f"{codec}.mkv"
        pl.apply(movie, cuts_path(movie), eng, output=out, verify="off", codec=codec)
        orig = decode(movie, 0, 2)[: RATE * 8]
        clean = decode(out, 1, 2)[: RATE * 8]
        assert media.best_lag(orig.mean(1), clean.mean(1)) == 0, codec


def test_mp4_output_and_replace_audio(movie, tmp_path):
    eng = fake_engines(SCRIPT)
    pl.analyze(movie, eng, stem_mode="off")
    out = tmp_path / "x.mp4"
    pl.apply(movie, cuts_path(movie), eng, output=out, verify="off", replace_audio=True)
    streams = media.audio_streams(media.probe(out))
    assert len(streams) == 1 and streams[0]["codec"] == "aac"


def test_review_clips_html(movie, tmp_path):
    eng = fake_engines(SCRIPT)
    pl.analyze(movie, eng, stem_mode="off")
    pl.review_cmd(movie, clips=tmp_path / "clips", show=False)
    files = sorted(p.name for p in (tmp_path / "clips").iterdir())
    assert "review.html" in files and len([f for f in files if f.endswith(".wav")]) == 4


def test_refine_pass_finds_word_only_audible_without_music(tmp_path, cache):
    """A cuss word Whisper mis-heard in the full mix (low prob) is re-checked on the isolated voice."""
    script = [dict(w) for w in SCRIPT]
    heard_badly = [dict(w) for w in script]
    for w in heard_badly:
        if w["word"] == "shit":
            w["word"], w["prob"] = "ship", 0.3            # main pass: wrong word, low confidence
    audio, _, _ = tone_audio(script, DUR)
    movie = write_movie(tmp_path / "m.mkv", audio)
    eng = fake_engines(heard_badly)

    class Refine:
        """Main pass hears 'ship'; the voice-only re-listen hears the real word."""
        def __init__(self):
            self.n = 0
            self.model_name, self.key, self.language = "fake", "fake", "en"

        def transcribe(self, audio16k, offset=0.0, vad=None, language=None, prompt=True):
            self.n += 1
            src = script if vad is False else heard_badly
            from conftest import ToneTranscriber
            return ToneTranscriber(src).transcribe(audio16k, offset)

    eng._asr = Refine()
    rv = pl.analyze(movie, eng, stem_mode="refine")
    shit = [d for d in rv["detections"] if d["text"] == "shit"]
    assert shit and "voice-pass" in shit[0]["flags"] and not shit[0].get("suggest")
    c = load_cuts(cuts_path(movie))
    assert any(r["start"] <= 20.0 - c["origin"] <= r["end"] for r in c["ranges"])


def test_voice_pass_corrects_an_unsure_main_pass_even_at_low_confidence(tmp_path, cache):
    """Regression (real run, seed 3): main heard 'For' at 40%, voice-only heard 'Fuck' at 2% -> remove it.
    The confident next word ('this,' 99%) touches the cuss word's end in the main pass; that must
    not count as the main pass having heard something else there."""
    script = [dict(w) for w in SCRIPT]
    main = []
    for w in script:
        if w["word"] == "fuck":
            main.append(dict(w, word="For", start=10.12, end=10.18, prob=0.4))
        elif w["word"] == "this" and w["start"] > 9:      # real run: touched the cuss word by 0.04 s
            main.append(dict(w, start=10.36, end=10.78, prob=0.99))
        else:
            main.append(dict(w))
    voice = [dict(w, prob=0.022) if w["word"] == "fuck" else dict(w) for w in script]
    audio, _, _ = tone_audio(script, DUR)
    movie = write_movie(tmp_path / "m.mkv", audio)

    class Asr:
        model_name, key, language = "fake", "fake", "en"

        def transcribe(self, audio16k, offset=0.0, vad=None, language=None, prompt=True):
            src = voice if vad is False else main
            dur = len(audio16k) / 16000
            return [w for w in src if offset <= w["start"] and w["end"] <= offset + dur], "en"

    eng = fake_engines(script)
    eng._asr = Asr()
    rv = pl.analyze(movie, eng, stem_mode="refine")
    d = [x for x in rv["detections"] if x["text"] == "fuck"][0]
    assert "voice-pass" in d["flags"] and not d.get("suggest")
    c = load_cuts(cuts_path(movie))
    assert any(r["start"] <= 10.1 - c["origin"] <= r["end"] for r in c["ranges"])


def test_voice_pass_never_overrules_a_confident_main_pass(tmp_path, cache):
    """Voice-only listen 'hears' a cuss word where the main pass was sure of another word -> 'maybe'."""
    audio, _, _ = tone_audio(SCRIPT, DUR)
    movie = write_movie(tmp_path / "m.mkv", audio)
    main = [dict(w) for w in SCRIPT] + [{"word": "Chicago", "start": 15.0, "end": 15.4, "prob": 0.97},
                                        {"word": "maybe", "start": 15.5, "end": 15.8, "prob": 0.3}]
    voice = [dict(w) for w in SCRIPT] + [{"word": "shit", "start": 15.05, "end": 15.35, "prob": 0.8}]

    class Asr:
        model_name, key, language = "fake", "fake", "en"

        def transcribe(self, audio16k, offset=0.0, vad=None, language=None, prompt=True):
            src = voice if vad is False else main
            dur = len(audio16k) / 16000
            return [w for w in src if offset <= w["start"] and w["end"] <= offset + dur], "en"

    eng = fake_engines(SCRIPT)
    eng._asr = Asr()
    pl.analyze(movie, eng, stem_mode="refine")
    rows = pl.review_cmd(movie, show=False)
    maybe = [r for r in rows if r["text"] == "shit" and r["start"] == 15.05][0]
    assert maybe["pending"] and not maybe["active"] and "unconfirmed" in maybe["flags"]
    def covered(t):
        c = load_cuts(cuts_path(movie))
        return any(r["start"] <= t - c["origin"] <= r["end"] for r in c["ranges"])
    assert not covered(15.1)
    pl.review_cmd(movie, restore=[maybe["n"]], show=False)          # user listens and accepts
    assert covered(15.1)


def test_batch_skips_done_and_reports(tmp_path, cache):
    lib = tmp_path / "Movies"
    (lib / "A (2001)").mkdir(parents=True)
    audio, _, _ = tone_audio(SCRIPT, DUR)
    m = write_movie(lib / "A (2001)" / "A (2001).mkv", audio)
    (lib / "A (2001)" / "A (2001)-trailer.mkv").write_bytes(m.read_bytes())
    eng = fake_engines(SCRIPT)
    res = pl.batch(lib, eng, in_place=True, analyze_opts={"stem_mode": "off"},
                   apply_opts={"verify": "fast", "codec": "flac"})
    assert len(res) == 1 and res[0][1].startswith("ok")
    res = pl.batch(lib, eng, in_place=True, analyze_opts={"stem_mode": "off"}, apply_opts={"verify": "fast"})
    assert res[0][1] == "skipped"


def test_disk_space_check(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("u", (), {"free": 10})())
    with pytest.raises(SystemExit):
        media.check_space(tmp_path, 1e9, "test")


def test_choose_stream_rules():
    s = [{"order": 0, "clean": False, "title": "Commentary", "language": "eng", "default": False},
         {"order": 1, "clean": False, "title": "", "language": "fre", "default": True},
         {"order": 2, "clean": False, "title": "", "language": "eng", "default": False},
         {"order": 3, "clean": True, "title": "Clean (no profanity)", "language": "eng", "default": False}]
    assert media.choose_stream(s)["order"] == 1
    assert media.choose_stream(s, language="en")["order"] == 2
    assert media.choose_stream(s, order=0)["order"] == 0
    with pytest.raises(SystemExit):
        media.choose_stream(s, order=9)


def test_channel_groups():
    st = {"channels": 6, "layout": "5.1(side)"}
    assert media.channel_groups(st) == [(0, 1), (2,)]
    assert media.channel_groups(st, surrounds=True) == [(0, 1), (2,), (4, 5)]
    assert media.dialogue_channels(st) == [2]
    assert media.channel_groups({"channels": 2, "layout": "stereo"}) == [(0, 1)]
    assert media.channel_groups({"channels": 1, "layout": "mono"}) == [(0,)]
    assert media.channel_groups({"channels": 8, "layout": ""}, True) == [(0, 1), (2,), (4, 5), (6, 7)]
    assert media.channel_groups({"channels": 8, "layout": "7.1"}) == [(0, 1), (2,)]


def test_v1_cuts_file_still_loads(tmp_path):
    p = tmp_path / "old.cuts.json"
    p.write_text(json.dumps({"title": "x", "anchor": {"text": "hello there"}, "cusses": [{"start": 1, "end": 2}]}))
    c = load_cuts(p)
    assert c["anchors"][0]["text"] == "hello there" and c["ranges"][0]["level"] == "strong"


def test_review_needs_analyze(tmp_path):
    with pytest.raises(SystemExit):
        pl.review_cmd(tmp_path / "nothing.mkv", show=False)
    assert not review_path(tmp_path / "nothing.mkv").exists()
