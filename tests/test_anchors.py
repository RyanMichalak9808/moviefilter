import pytest

from moviefilter.anchors import TimeMap, locate, nearest_known, pick_anchors, resolve
from moviefilter.wordlist import Matcher

LINES = [
    "welcome back to the harbor the ships came in early this morning",
    "captain reynolds wants the cargo moved before the sunset tonight",
    "the train to chicago leaves at a quarter past nine every day",
    "my grandmother used to bake fresh bread every single sunday morning",
    "remember to water the tomato plants in the garden after lunch",
    "the museum closes early on public holidays and some weekends too",
]


def film(runtime=3000.0, shift=0.0, scale=1.0, cut=None):
    """Words spread over a long 'film'. cut=(at, seconds) removes a chunk after `at` (different edit)."""
    words = []
    starts = [30, 400, 900, 1500, 2200, 2800]
    for line, s0 in zip(LINES, starts):
        for i, w in enumerate(line.split()):
            t = s0 + i * 0.45
            if cut and t > cut[0]:
                t -= cut[1]
            t = t * scale + shift
            words.append({"word": w, "start": t, "end": t + 0.35, "prob": 0.95})
    return words


def words_fn(words):
    return lambda a, b: [w for w in words if a <= w["start"] <= b]


def test_pick_anchors_spread_and_unique():
    ws = film()
    anchors = pick_anchors(ws, Matcher().find(ws), 3000.0)
    assert len(anchors) == 3
    assert anchors[0]["time"] == pytest.approx(30.0)
    assert anchors[1]["time"] > 1000 and anchors[2]["time"] > 2000


def test_pick_anchors_skips_profanity():
    ws = film()
    ws[2]["word"] = "fucking"
    anchors = pick_anchors(ws, Matcher().find(ws), 3000.0)
    assert "fucking" not in anchors[0]["text"]


def test_locate_fuzzy():
    ws = film()
    t, s = locate(ws, "the train to chicago leaves at quarter past")
    assert t == pytest.approx(900.0) and s > 0.8
    t, s = locate(ws, "completely unrelated words that never appear here")
    assert t is None


def cuts_from(ws, runtime=3000.0):
    anchors = pick_anchors(ws, [], runtime)
    t0 = anchors[0]["time"]
    return {"anchors": [{"text": a["text"], "t": a["time"] - t0} for a in anchors], "origin": t0, "runtime": runtime}


def test_resolve_offset():
    cuts = cuts_from(film())
    tm, rep = resolve(cuts, words_fn(film(shift=12.5)), 3012.5)
    assert tm(0) == pytest.approx(42.5, abs=0.01)
    assert tm(2000) == pytest.approx(2042.5, abs=0.05)
    assert rep["warnings"] == []            # a longer intro alone is not worth a warning


def test_resolve_pal_speedup():
    s = 23.976 / 25
    cuts = cuts_from(film())
    tm, rep = resolve(cuts, words_fn(film(scale=s, shift=3)), 3000 * s + 3)
    assert tm(2000) == pytest.approx((30 + 2000) * s + 3, abs=0.1)
    assert any("PAL" in w and "faster" in w for w in rep["warnings"])
    assert not any("longer after" in w for w in rep["warnings"])


def test_resolve_different_cut_warns_and_interpolates():
    cuts = cuts_from(film())
    other = film(cut=(1000, 20.0))              # 20 s removed between anchor 1 and 2
    tm, rep = resolve(cuts, words_fn(other), 2980.0)
    assert any("different cut" in w for w in rep["warnings"])
    assert tm(2200 - 30) == pytest.approx(2180.0, abs=0.1)      # after the cut: exact again


def test_resolve_missing_first_anchor():
    cuts = cuts_from(film())
    with pytest.raises(SystemExit):
        resolve(cuts, lambda a, b: [], 3000.0)


def test_timemap_extrapolates():
    tm = TimeMap([(0, 10), (100, 110)])
    assert tm(-5) == pytest.approx(5)
    assert tm(200) == pytest.approx(210)
    assert TimeMap([(0, 10)], slope=1.04)(100) == pytest.approx(114)


def test_nearest_known():
    assert nearest_known(25 / 23.976)[0] == pytest.approx(25 / 23.976)
    assert nearest_known(1.02)[0] is None
