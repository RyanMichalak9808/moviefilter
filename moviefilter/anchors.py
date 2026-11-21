"""Anchors: spoken lines that tie a shared cuts file to any copy of the same film.

analyze picks several anchors (start, middle, end). apply finds them in the target
copy and builds a piecewise-linear time map, which handles:
  * different intro/logo lengths      (offset)
  * PAL speed-up / 24 vs 23.976 fps   (constant slope)
  * small edit differences            (piecewise, with a warning)
"""
import difflib

from .util import fmt_time, log
from .wordlist import norm

ANCHOR_WORDS = 8
MIN_SCORE = 0.75
# slope = (time in this copy) / (time in the copy the cuts file was made from)
KNOWN_SLOPES = {
    1.0: "same speed",
    23.976 / 25: "this copy plays ~4% faster (PAL 25 fps vs 23.976)",
    25 / 23.976: "this copy plays ~4% slower (23.976 fps vs PAL 25)",
    24 / 25: "this copy plays ~4% faster (PAL 25 fps vs 24)",
    25 / 24: "this copy plays ~4% slower (24 fps vs PAL 25)",
    23.976 / 24: "this copy plays 0.1% faster (24 vs 23.976 fps)",
    24 / 23.976: "this copy plays 0.1% slower (23.976 vs 24 fps)",
}


def _toks(words):
    return [norm(w["word"]) for w in words]


def _candidates(words, excluded, n=ANCHOR_WORDS):
    """Runs of n words: confident, continuous speech, real words, no detected cussing."""
    toks = _toks(words)
    for i in range(len(words) - n + 1):
        run = words[i:i + n]
        if any(j in excluded for j in range(i, i + n)):
            continue
        if any(run[k + 1]["start"] - run[k]["end"] > 1.2 for k in range(n - 1)):
            continue
        t = toks[i:i + n]
        if sum(len(x) for x in t) < 3 * n or not all(t):
            continue
        p = sum(w.get("prob", 1.0) for w in run) / n
        if p < 0.8:
            continue
        yield i, p


def pick_anchors(words, detections, runtime, count=3):
    """Pick up to `count` anchors spread over the film. Returns [{text, time}] sorted by time."""
    excluded = {j for d in detections for j in range(d.i0, d.i1)}
    cands = list(_candidates(words, excluded))
    if not cands:
        return []
    texts = {}
    for i, _ in cands:
        key = " ".join(_toks(words[i:i + ANCHOR_WORDS]))
        texts[key] = texts.get(key, 0) + 1

    def ok(i):
        return texts[" ".join(_toks(words[i:i + ANCHOR_WORDS]))] == 1      # unique line

    chosen = []
    first = next(((i, p) for i, p in cands if ok(i)), None)
    if first is None:
        return []
    chosen.append(first[0])
    targets = [runtime * f for f in ((0.5, 0.92) if count >= 3 else (0.92,))][: count - 1]
    for tgt in targets:
        pool = [(i, p) for i, p in cands if ok(i) and abs(words[i]["start"] - tgt) < max(300, 0.1 * runtime)
                and words[i]["start"] > words[chosen[-1]]["start"] + 60]
        if pool:
            best = max(pool, key=lambda ip: ip[1] - abs(words[ip[0]]["start"] - tgt) / runtime)
            chosen.append(best[0])
    return [{"text": " ".join(w["word"] for w in words[i:i + ANCHOR_WORDS]), "time": words[i]["start"]}
            for i in chosen]


def locate(words, text, expect=None):
    """Fuzzy-find `text` in `words`. Returns (time, score) or (None, best_score)."""
    target = " ".join(t for t in (norm(x) for x in text.split()) if t)
    n = len(target.split())
    toks = _toks(words)
    best, best_i = 0.0, None
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(target)
    for i in range(max(1, len(toks) - n + 1)):
        sm.set_seq1(" ".join(toks[i:i + n]))
        if sm.real_quick_ratio() < best - 1e-9:
            continue
        s = sm.ratio()
        closer = expect is not None and best_i is not None and abs(s - best) < 0.02 and \
            abs(words[i]["start"] - expect) < abs(words[best_i]["start"] - expect)
        if s > best + 0.02 or closer:
            best, best_i = max(s, best), i
    if best_i is None or best < MIN_SCORE:
        return None, best
    return words[best_i]["start"], best


def nearest_known(slope, tol=0.0025):
    k = min(KNOWN_SLOPES, key=lambda s: abs(s - slope))
    return (k, KNOWN_SLOPES[k]) if abs(k - slope) <= tol else (None, None)


class TimeMap:
    """Piecewise-linear map from cuts-file time (seconds after anchor 0) to target clock."""

    def __init__(self, points, slope=1.0):
        self.points = sorted(points)
        self.slope = slope

    def __call__(self, r):
        p = self.points
        if len(p) == 1:
            return p[0][1] + (r - p[0][0]) * self.slope
        if r <= p[0][0]:
            (x0, y0), (x1, y1) = p[0], p[1]
        elif r >= p[-1][0]:
            (x0, y0), (x1, y1) = p[-2], p[-1]
        else:
            for k in range(len(p) - 1):
                if p[k][0] <= r <= p[k + 1][0]:
                    (x0, y0), (x1, y1) = p[k], p[k + 1]
                    break
        return y0 + (r - x0) * (y1 - y0) / (x1 - x0)

    def segment_slopes(self):
        p = self.points
        return [(p[k + 1][1] - p[k][1]) / (p[k + 1][0] - p[k][0]) for k in range(len(p) - 1)]


def resolve(cuts, words_in, runtime):
    """Find the cuts file's anchors in this copy.

    words_in(start, end) -> transcript words for that span of this copy.
    Returns (TimeMap, report dict). Raises SystemExit if anchor 0 can't be found.
    """
    anchors = cuts["anchors"]
    src_runtime = cuts.get("runtime") or runtime
    ratio = runtime / src_runtime if src_runtime else 1.0
    guess, why = nearest_known(ratio, tol=0.006)
    slope = guess or 1.0
    origin = cuts.get("origin")
    warnings = []

    # anchor 0: the creator's copy had it at `origin`; intros can differ by minutes.
    if origin is not None:
        lo, hi = max(0.0, origin * slope - 600), min(runtime, origin * slope + 600)
    else:
        lo, hi = 0.0, min(runtime, 1200.0)
    t0, score = locate(words_in(lo, hi), anchors[0]["text"], expect=(origin or 0) * slope)
    if t0 is None and (lo > 0 or hi < runtime):
        log("  anchor 1 not near the expected spot, searching the first 30 minutes ...")
        t0, score = locate(words_in(0.0, min(runtime, 1800.0)), anchors[0]["text"])
    if t0 is None:
        raise SystemExit(f"Could not find the first anchor line in this copy (best match {score:.2f}):\n"
                         f"    {anchors[0]['text']!r}\n"
                         "This may be a different cut/language, or a poor transcription. "
                         "Pass --anchor-time H:MM:SS to set it by hand.")
    points = [(0.0, t0)]
    found = [{"text": anchors[0]["text"], "expected": None, "found": round(t0, 3), "score": round(score, 3)}]
    for a in anchors[1:]:
        exp = t0 + a["t"] * slope
        tol = max(120.0, 0.02 * a["t"])
        t, sc = locate(words_in(exp - tol, exp + tol), a["text"], expect=exp)
        if t is None:
            t, sc = locate(words_in(exp - 600, exp + 600), a["text"], expect=exp)
        found.append({"text": a["text"], "expected": round(exp, 3), "found": None if t is None else round(t, 3),
                      "score": round(sc, 3)})
        if t is None:
            warnings.append(f"anchor at +{fmt_time(a['t'])} not found; times past it are extrapolated")
        else:
            points.append((a["t"], t))

    tm = TimeMap(points, slope=slope)
    segs = tm.segment_slopes()
    if segs:
        tm.slope = segs[-1]
        for k, s in enumerate(segs):
            known, label = nearest_known(s)
            if known is None:
                drift = (s - 1) * (points[k + 1][0] - points[k][0])
                warnings.append(
                    f"between anchors {k + 1} and {k + 2} this copy runs {drift:+.1f}s different "
                    f"(slope {s:.5f}) - probably a different cut; times there are interpolated")
            elif known != 1.0:
                warnings.append(f"speed difference detected: {label}; times scaled to match")
    elif guess and guess != 1.0:
        warnings.append(f"runtime suggests {why}; only one anchor, so scaling is a guess")
    if origin is not None and cuts.get("runtime"):
        # expected end of this copy if only the intro length and speed differ
        expected = t0 + (src_runtime - origin) * tm.slope
        if abs(runtime - expected) > 5.0:
            warnings.append(f"this copy is {runtime - expected:+.0f}s longer after the first anchor than "
                            f"expected (different cut, extra credits or bonus scene?) - check the result")
    return tm, {"anchors": found, "slopes": [round(s, 6) for s in segs], "warnings": warnings}
