"""High-level operations: analyze, review, apply, batch, doctor."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from . import __version__, media
from .anchors import TimeMap, locate, pick_anchors, resolve
from .asr import Transcriber, default_model, transcribe_audio, transcribe_span
from .cuts import (active_items, build_cuts, cuts_path, load_cuts, load_review, review_path,
                   save_review, to_ranges)
from .util import Progress, fmt_time, log, parse_range, parse_time, read_json, write_json
from .wordlist import DEFAULT_LEVELS, LEVELS, Matcher

PIPELINE_VERSION = 1          # bump when patch output changes (invalidates cached patches)
REFINE_PROB = 0.5             # words below this confidence get a second, voice-only listen
REFINE_PAD = 2.5
MAIN_SURE = 0.6               # main-pass word confidence that outranks a voice-pass guess


# ------------------------------------------------------------------ shared plumbing
class Engines:
    """Lazily-built heavy objects, shared across files in batch mode."""

    def __init__(self, model=None, language="en", device="auto", vad=True, demucs_model="htdemucs",
                 shifts=1, wordlist=None, extra_words=(), allow=()):
        self.model, self.language, self.device, self.vad = model, language, device, vad
        self.demucs_model, self.shifts = demucs_model, shifts
        self.wordlist, self.extra_words, self.allow = wordlist, extra_words, allow
        self._asr, self._sep = None, None

    @property
    def asr(self):
        if self._asr is None:
            model = self.model or default_model(self.language)
            self._asr = Transcriber(model, self.language, device=self.device, vad=self.vad)
        return self._asr

    @property
    def sep(self):
        if self._sep is None:
            from .separate import DemucsSeparator
            self._sep = DemucsSeparator(self.demucs_model, device=self.device, shifts=self.shifts)
        return self._sep

    def matcher(self, language, levels=LEVELS):
        return Matcher(languages=[language, "en"], levels=levels, wordlist=self.wordlist,
                       extra_words=self.extra_words, allow=self.allow)


def workdir_for(movie, base=None):
    p = Path(movie).resolve()
    st = p.stat()
    h = hashlib.sha1(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:10]
    root = Path(base or os.environ.get("MOVIEFILTER_CACHE") or Path.home() / ".cache" / "moviefilter")
    safe = re.sub(r"[^\w.-]+", "_", p.stem)[:40]
    d = root / f"{safe}-{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def open_stream(movie, order=None, language=None):
    media.require_tools()
    info = media.probe(movie)
    streams = media.audio_streams(info)
    stream = media.choose_stream(streams, order, language)
    log(f"  audio track {stream['order']}: {stream['codec']} {stream['channels']}ch {stream['rate']} Hz "
        f"[{stream['language']}] {stream['title']!r}")
    return info, streams, stream


def get_audio16k(movie, stream, wd, space_check=True):
    f = Path(wd) / f"audio16k_a{stream['order']}.s16"
    if not f.exists():
        if space_check:
            media.check_space(wd, stream["duration"] * 32000, "16 kHz analysis audio")
        log("  decoding audio for analysis ...")
        media.decode_16k(movie, stream, f)
    return media.Audio16k(f)


def _overlaps(a, b, pad=0.0):
    return a["start"] < b["end"] + pad and a["end"] > b["start"] - pad


def _aligned(a, b):
    """Same spoken word: overlap covers at least half of the shorter of the two
    (a neighbouring word that merely touches doesn't count)."""
    ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return ov > 0 and ov >= 0.5 * min(a["end"] - a["start"], b["end"] - b["start"])


# ------------------------------------------------------------------ analyze
def analyze(movie, eng, levels=DEFAULT_LEVELS, stem_mode="refine", refine_max=900.0, audio_track=None,
            anchor_text=None, anchor_time=None, workdir=None, space_check=True):
    movie = Path(movie)
    log(f"Analyzing {movie.name}")
    info, _, stream = open_stream(movie, audio_track, None if eng.language in (None, "auto") else eng.language)
    wd = workdir_for(movie, workdir)
    audio = get_audio16k(movie, stream, wd, space_check)
    words, lang = transcribe_audio(audio, eng.asr, wd / "asr")
    write_json(wd / "words.json", words)
    matcher = eng.matcher(lang)
    dets = [d.to_dict() for d in matcher.find(words)]
    for d in dets:
        d["source"] = "main"

    extra = []
    if stem_mode == "refine":
        extra = _refine_pass(audio, words, dets, eng, matcher, lang, wd, refine_max)
    elif stem_mode == "full":
        extra = _full_voice_pass(audio, eng, matcher, lang, wd)
    for d in extra:
        if not any(_overlaps(d, x, 0.1) for x in dets):
            d["flags"] = sorted(set(d["flags"]) | {"voice-pass"})
            d["source"] = "voice"
            # The main pass was sure it heard a different word here: the voice-only guess is
            # probably a mishearing ("Chicago" -> "shit"). Keep it as a suggestion (shown in
            # review, not removed). If the main pass was itself unsure ("For" 40% where the line
            # says "Fuck"), the voice pass is allowed to correct it - its own confidence is not a
            # useful signal on isolated voice, real hits measured as low as 2%.
            if any(_aligned(d, w) and w.get("prob", 1) >= MAIN_SURE for w in words):
                d["flags"].append("unconfirmed")
                d["suggest"] = True
            dets.append(d)
    dets.sort(key=lambda d: d["start"])
    for d in dets:
        d["id"] = f"{d['start']:.2f}"

    det_objs = matcher.find(words)       # index-based exclusion for anchor picking
    anchors = pick_anchors(words, det_objs, audio.duration)
    if anchor_text:
        t = parse_time(anchor_time) if anchor_time else locate(words, anchor_text)[0]
        if t is None:
            raise SystemExit(f"--anchor-text {anchor_text!r} not found in the transcript.")
        anchors = [{"text": anchor_text, "time": t}] + [a for a in anchors if a["time"] > t + 60]
    if not anchors:
        raise SystemExit("No usable anchor line found (too little clear speech?). Use --anchor-text.")

    review = load_review(movie)
    review.update({"version": 1, "movie": movie.name, "language": lang, "runtime": audio.duration,
                   "stream_order": stream["order"], "anchors": anchors, "levels": list(levels),
                   "model": eng.asr.model_name, "detections": dets})
    save_review(movie, review)
    cuts = write_cuts(movie, review)

    by = {}
    for d in dets:
        by[d["level"]] = by.get(d["level"], 0) + 1
    flagged = sum(1 for d in dets if d["flags"])
    log(f"  language {lang}; {len(words)} words; detections: "
        + (", ".join(f"{k} {v}" for k, v in sorted(by.items())) or "none"))
    log(f"  {len(cuts['ranges'])} ranges in {cuts_path(movie).name} (levels: {', '.join(levels)}); "
        f"{flagged} flagged for review")
    log("  anchors: " + "; ".join(f"{fmt_time(a['time'])} {a['text']!r}" for a in anchors))
    return review


def _refine_pass(audio, words, dets, eng, matcher, lang, wd, cap):
    """Re-listen to low-confidence passages on the isolated voice (music/effects removed)."""
    from .separate import voice_16k
    spots = [(w["prob"], w["start"], w["end"]) for w in words if w.get("prob", 1) < REFINE_PROB]
    spots += [(d["prob"], d["start"], d["end"]) for d in dets if d["flags"]]
    spots.sort()
    regions, total = [], 0.0
    for _, s, e in spots:
        a, b = max(0.0, s - REFINE_PAD), min(audio.duration, e + REFINE_PAD)
        if any(a < r[1] and b > r[0] for r in regions):
            continue
        if total + (b - a) > cap:
            break
        regions.append([a, b])
        total += b - a
    if not regions:
        return []
    # merge neighbours so Demucs gets context
    regions.sort()
    merged = [regions[0]]
    for a, b in regions[1:]:
        if a <= merged[-1][1] + 2:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    log(f"  voice-only re-listen of {len(merged)} unclear passages ({total / 60:.1f} min) ...")
    cache = wd / "refine"
    cache.mkdir(exist_ok=True)
    out = []
    prog = Progress("re-listen", len(merged))
    for a, b in merged:
        f = cache / f"{a:.2f}-{b:.2f}-{eng.asr.key}.json"
        ws = read_json(f)
        if ws is None:
            v = voice_16k(eng.sep, audio.read(a, b))
            ws, _ = eng.asr.transcribe(v, offset=a, vad=False, language=lang)
            write_json(f, ws)
        out += [d.to_dict() for d in matcher.find(ws)]
        prog.update()
    prog.done()
    return out


def _full_voice_pass(audio, eng, matcher, lang, wd):
    """Separate the voice for the entire film, then transcribe that too (slow, most thorough)."""
    from .separate import voice_16k
    vf = wd / "voice16k.s16"
    if not vf.exists():
        log("  isolating the voice track for the whole film (slow on CPU) ...")
        tmp = vf.with_name(vf.name + ".part")
        block = 60.0
        n = int(np.ceil(audio.duration / block))
        prog = Progress("voice isolation", n)
        with open(tmp, "wb") as fh:
            for k in range(n):
                a, b = k * block, min(audio.duration, (k + 1) * block)
                x = audio.read(a, b)
                v = voice_16k(eng.sep, x) if len(x) else x
                fh.write((np.clip(v, -1, 1) * 32767).astype(np.int16).tobytes())
                prog.update()
        prog.done()
        tmp.replace(vf)
    vwords, _ = transcribe_audio(media.Audio16k(vf), eng.asr, wd / "asr-voice", label="transcribe voice")
    return [d.to_dict() for d in matcher.find(vwords)]


def write_cuts(movie, review):
    items = active_items(review, review.get("levels"))
    ranges = to_ranges(items)
    cuts = build_cuts(Path(movie).stem, review["language"], review["runtime"], review["anchors"], ranges)
    write_json(cuts_path(movie), cuts)
    return cuts


# ------------------------------------------------------------------ review
def review_rows(review):
    rows = [dict(d, row_kind="det") for d in review["detections"]]
    rows += [dict(a, row_kind="add", text="(manual)", level=a.get("level", "manual"), kind="manual",
                  prob=1.0, flags=[], context="") for a in review["edits"]["add"]]
    rows.sort(key=lambda r: r["start"])
    drops = [d["t"] for d in review["edits"]["drop"]]
    accepts = [a["t"] for a in review["edits"].get("accept", [])]
    levels = review.get("levels") or list(LEVELS)
    for i, r in enumerate(rows, 1):
        r["n"] = i
        r["dropped"] = any(r["start"] - 0.05 <= t <= r["end"] + 0.05 for t in drops)
        r["pending"] = bool(r.get("suggest")) and not any(r["start"] - 0.05 <= t <= r["end"] + 0.05 for t in accepts)
        r["active"] = (r["row_kind"] == "add") or (not r["dropped"] and not r["pending"] and r["level"] in levels)
    return rows


def _row_status(r):
    if r["dropped"]:
        return "DROPPED"
    if r["pending"]:
        return "maybe"
    return "muted" if r["active"] else "off"


def review_cmd(movie, drop=(), restore=(), add=(), add_level="manual", levels=None, flagged_only=False,
               clips=None, workdir=None, show=True):
    movie = Path(movie)
    review = load_review(movie)
    if not review["detections"] and not review["edits"]["add"] and not review_path(movie).exists():
        raise SystemExit(f"No review data for {movie.name}; run `analyze` first.")
    rows = review_rows(review)
    by_n = {r["n"]: r for r in rows}
    changed = False
    for n in drop:
        r = by_n.get(n) or _bad_row(n)
        if r["row_kind"] == "add":
            review["edits"]["add"] = [a for a in review["edits"]["add"]
                                      if not (a["start"] == r["start"] and a["end"] == r["end"])]
        else:
            review["edits"]["drop"].append({"t": round((r["start"] + r["end"]) / 2, 3), "text": r["text"]})
        changed = True
    for n in restore:
        r = by_n.get(n) or _bad_row(n)
        review["edits"]["drop"] = [d for d in review["edits"]["drop"]
                                   if not (r["start"] - 0.05 <= d["t"] <= r["end"] + 0.05)]
        if r.get("suggest"):          # accept an unconfirmed voice-pass suggestion
            review["edits"].setdefault("accept", []).append({"t": round((r["start"] + r["end"]) / 2, 3)})
        changed = True
    for spec in add:
        a, b = parse_range(spec)
        review["edits"]["add"].append({"start": a, "end": b, "level": add_level})
        changed = True
    if levels:
        review["levels"] = list(levels)
        changed = True
    if changed:
        save_review(movie, review)
        cuts = write_cuts(movie, review)
        log(f"Saved. {len(cuts['ranges'])} ranges in {cuts_path(movie).name}")
        rows = review_rows(review)
    if clips:
        export_clips(movie, review, rows, Path(clips), workdir)
    if show:
        print_rows(rows, flagged_only)
    return rows


def _bad_row(n):
    raise SystemExit(f"No row #{n}. Run `review` without options to see the list.")


def print_rows(rows, flagged_only=False):
    print(f"{'#':>4}  {'time':>11}  {'level':9} {'kind':7} {'conf':>4}  {'status':8} flags / context")
    for r in rows:
        if flagged_only and not r["flags"] and r["row_kind"] != "add":
            continue
        status = _row_status(r)
        ctx = r.get("context") or f"{fmt_time(r['start'])}-{fmt_time(r['end'])}"
        flags = (",".join(r["flags"]) + " | ") if r["flags"] else ""
        print(f"{r['n']:>4}  {fmt_time(r['start']):>11}  {r['level']:9} {r['kind']:7} {r['prob']:>4.2f}  "
              f"{status:8} {flags}{ctx}")
    print("\nEdit with: review MOVIE --drop N [N ...] | --restore N (also accepts a 'maybe') | "
          "--add H:MM:SS.s-H:MM:SS.s | --clips DIR (listen)")


def export_clips(movie, review, rows, out, workdir=None):
    """Write a short WAV around every row + an HTML page to listen to them in a browser."""
    out.mkdir(parents=True, exist_ok=True)
    info = media.probe(movie)
    stream = media.choose_stream(media.audio_streams(info), review.get("stream_order"))
    audio = get_audio16k(movie, stream, workdir_for(movie, workdir))
    html = ["<!doctype html><meta charset=utf-8><title>moviefilter review</title>",
            "<style>body{font:14px system-ui;margin:16px}td{padding:4px 8px;border-bottom:1px solid #ddd}"
            ".off{opacity:.5}</style><h1>Review: " + movie.name + "</h1><table>",
            "<tr><th>#</th><th>time</th><th>level</th><th>status</th><th>context</th><th>listen</th></tr>"]
    for r in rows:
        a, b = max(0.0, r["start"] - 2.0), r["end"] + 2.0
        x = audio.read(a, b)
        name = f"{r['n']:04d}_{fmt_time(r['start']).replace(':', '-')}_{r['level']}.wav"
        with wave.open(str(out / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        status = _row_status(r)
        ctx = (r.get("context") or "").replace("<", "&lt;")
        html.append(f"<tr class={'off' if not r['active'] else ''}><td>{r['n']}</td><td>{fmt_time(r['start'])}"
                    f"</td><td>{r['level']}</td><td>{status}</td><td>{ctx}</td>"
                    f"<td><audio controls preload=none src='{name}'></audio></td></tr>")
    html.append("</table><p>Drop false hits with: <code>review MOVIE --drop N</code></p>")
    (out / "review.html").write_text("\n".join(html), encoding="utf-8")
    log(f"  wrote {len(rows)} clips + review.html to {out}")


# ------------------------------------------------------------------ apply
def apply(movie, cuts_file, eng, output=None, in_place=False, backup=True, levels=None, audio_track=None,
          anchor_time=None, verify="full", surrounds=False, codec=None, bitrate=None,
          track_title="Clean (no profanity)", make_default=False, replace_audio=False, workdir=None,
          space_check=True, plex=None):
    from .separate import Cleaner, FADE, Target, plan_windows

    movie = Path(movie)
    if not output and not in_place:
        raise SystemExit("Give -o OUTPUT or --in-place.")
    log(f"Cleaning {movie.name}")
    cuts = load_cuts(cuts_file)
    info, streams, stream = open_stream(movie, audio_track, cuts.get("language"))
    if any(s["clean"] for s in streams):
        log("  file already has a clean track - it will be replaced")
    wd = workdir_for(movie, workdir)
    audio = get_audio16k(movie, stream, wd, space_check)
    runtime = audio.duration

    ranges = cuts["ranges"]
    if levels:
        ranges = [r for r in ranges if set(r.get("levels", [r["level"]])) & set(levels) or r["src"] == "manual"]
    tm, anchor_report = _time_map(movie, cuts, audio, eng, wd, anchor_time)
    for w in anchor_report["warnings"]:
        log(f"  WARNING: {w}")

    targets = []
    for i, r in enumerate(ranges):
        s, e = tm(r["start"]), tm(r["end"])
        cs, ce = tm(r["core"][0]), tm(r["core"][1])
        if e <= 0 or s >= runtime:
            continue
        targets.append(Target(f"r{i}", max(0.0, s), min(runtime, e), cs, ce, r["level"]))
    log(f"  {len(targets)} ranges to clean")

    rate, ch = stream["rate"], stream["channels"]
    windows = plan_windows(targets, runtime)
    groups = media.channel_groups(stream, surrounds)
    dialog = media.dialogue_channels(stream)
    codec_name, br = media.default_codec(ch)
    codec, bitrate = codec or codec_name, bitrate or br
    if space_check:
        win_bytes = sum(w.end - w.start for w in windows) * rate * ch * 4 * 0.5
        media.check_space(wd, win_bytes + runtime * 640_000 / 8, "patches + encoded track")
        out_dir = (Path(output).parent if output else movie.parent)
        media.check_space(out_dir, movie.stat().st_size + runtime * 640_000 / 8, "output file")

    matcher = eng.matcher(cuts.get("language") or "en")
    cleaner = Cleaner(eng.sep, stream, groups, dialog, transcriber=eng.asr if verify == "full" else None,
                      matcher=matcher, verify=verify, language=cuts.get("language"))
    settings = f"{PIPELINE_VERSION}|{eng.demucs_model}|{eng.shifts}|{verify}|{groups}|{rate}|{ch}|" \
               f"{eng.asr.key if verify == 'full' else ''}"
    patch_dir = wd / "patches"
    patch_dir.mkdir(exist_ok=True)

    # ---- pass 1: stream the audio, clean each window as soon as it's fully read
    jobs = []
    for w in windows:
        key = hashlib.sha1((settings + "|" + json.dumps(
            [round(w.start, 3), round(w.end, 3)] + [[round(t.start, 3), round(t.end, 3), round(t.core_start, 3),
                                                       round(t.core_end, 3)] for t in w.targets])).encode()).hexdigest()[:20]
        jobs.append({"win": w, "a": int(w.start * rate), "b": int(w.end * rate), "key": key,
                     "file": patch_dir / f"{key}.npz", "buf": []})
    todo = [j for j in jobs if not j["file"].exists()]
    reports = []
    if todo:
        log(f"  cleaning {len(todo)} windows ({len(jobs) - len(todo)} cached) ...")
        prog = Progress("clean", len(todo))
        pending = list(todo)
        for pos, block in media.iter_blocks(movie, stream):
            end = pos + len(block)
            for j in pending:
                if j["a"] < end and j["b"] > pos:
                    j["buf"].append(block[max(0, j["a"] - pos): min(len(block), j["b"] - pos)].copy())
            done = [j for j in pending if j["b"] <= end]
            for j in done:
                _process_job(j, cleaner, rate, FADE)
                pending.remove(j)
                prog.update()
            if not pending:
                break
        for j in pending:                # windows that run past the real end of the audio
            if j["buf"]:
                _process_job(j, cleaner, rate, FADE)
            prog.update()
        prog.done()
    for j in jobs:
        meta = j["file"].with_suffix(".json")
        if meta.exists():
            reports += json.loads(meta.read_text())

    # ---- pass 2: stream again, apply patches, encode
    clean_audio = wd / f"clean_a{stream['order']}.mka"
    _encode_with_patches(movie, stream, jobs, clean_audio, codec, bitrate)

    # ---- mux
    final = Path(output) if output else movie
    tmp_out = final.with_name(f".{final.stem}.moviefilter-tmp{final.suffix}")
    log("  muxing ...")
    media.mux(movie, clean_audio, tmp_out, stream, info, track_title, replace_audio, make_default)
    _check_output(tmp_out, info, replace_audio)
    if in_place and backup:
        bdir = movie.parent / ".moviefilter-backup"
        bdir.mkdir(exist_ok=True)
        shutil.move(str(movie), str(bdir / movie.name))
        log(f"  original moved to {bdir / movie.name}")
    os.replace(tmp_out, final)

    summary = _summarize(reports)
    report = {"movie": movie.name, "output": str(final), "version": __version__, "anchors": anchor_report,
              "summary": summary, "ranges": reports}
    write_json(final.with_suffix(".report.json") if not in_place else wd / "report.json", report)
    log(f"Done: {final}")
    log("  " + ", ".join(f"{k}: {v}" for k, v in summary.items()))
    for r in reports:
        if r["status"] == "ducked":
            log(f"  listen at {fmt_time(r['start'])}: voice could not be fully separated, range was ducked")
        elif r["status"] == "check":
            log(f"  listen at {fmt_time(r['start'])}: voice measured gone, but the re-check still heard the word")
    maybes = [d for d in (read_json(review_path(movie)) or {}).get("detections", []) if d.get("suggest")]
    if maybes:
        log(f"  {len(maybes)} unconfirmed 'maybe' detection(s) were not removed - see `review --flagged`")
    if plex:
        from .plex import refresh
        refresh(plex.get("url"), plex.get("token"), str(final.resolve()), plex.get("section"))
    return report


def _process_job(j, cleaner, rate, fade_s):
    audio = np.concatenate(j["buf"]) if len(j["buf"]) > 1 else j["buf"][0]
    j["buf"] = []
    win_t0 = j["a"] / rate
    targets = [t for t in j["win"].targets if t.start < win_t0 + len(audio) / rate]
    if not targets:
        np.savez(j["file"], idx=np.zeros((0, 2), dtype=np.int64))
        j["file"].with_suffix(".json").write_text("[]")
        return
    cleaned, ranges, reports = cleaner.process(win_t0, audio, targets)
    fade = int(fade_s * rate)
    arrays, idx = {}, []
    for k, (a, b) in enumerate(ranges):
        a0, b0 = max(0, a - fade), min(len(cleaned), b + fade)
        arrays[f"p{k}"] = cleaned[a0:b0].astype(np.float32)
        idx.append([j["a"] + a0, j["a"] + b0])
    tmp = j["file"].with_name(j["file"].stem + ".part.npz")
    np.savez(tmp, idx=np.array(idx, dtype=np.int64).reshape(-1, 2), **arrays)
    os.replace(tmp, j["file"])
    j["file"].with_suffix(".json").write_text(json.dumps(reports))


def _encode_with_patches(movie, stream, jobs, out_file, codec, bitrate):
    patches = []
    for j in jobs:
        with np.load(j["file"]) as z:
            for k, (a, b) in enumerate(z["idx"]):
                patches.append((int(a), int(b), j["file"], k))
    patches.sort()
    # Encoders add "priming" samples at the start (AAC ~1024). Drop that many from the input so the
    # clean track lines up sample-for-sample with the original track (no lip-sync drift).
    skip = media.codec_delay(codec, bitrate, stream["rate"], stream["channels"], stream["layout"],
                             Path(out_file).parent)
    log(f"  writing clean track ({codec}, encoder delay {skip} samples compensated) "
        f"with {len(patches)} patches ...")
    enc = media.encoder(out_file, stream["rate"], stream["channels"], stream["layout"], codec, bitrate)
    prog = Progress("encode", stream["duration"] or 1)
    pi, cache = 0, {}
    try:
        for pos, block in media.iter_blocks(movie, stream, block_sec=5.0):
            end = pos + len(block)
            out = None
            while pi < len(patches) and patches[pi][1] <= pos:
                pi += 1
            k = pi
            while k < len(patches) and patches[k][0] < end:
                a, b, f, n = patches[k]
                if (f, n) not in cache:
                    with np.load(f) as z:
                        cache[(f, n)] = z[f"p{n}"]
                data = cache[(f, n)]
                s, e = max(a, pos), min(b, end)
                if out is None:
                    out = block.copy()
                out[s - pos:e - pos] = data[s - a:e - a]
                if b <= end:
                    cache.pop((f, n), None)
                k += 1
            data = block if out is None else out
            if skip:
                cut = min(skip, len(data))
                data, skip = data[cut:], skip - cut
            enc.stdin.write(data.astype(np.float32).tobytes())
            prog.update(len(block) / stream["rate"])
    finally:
        enc.stdin.close()
        rc = enc.wait()
    prog.done()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed ({rc})")


def _time_map(movie, cuts, audio, eng, wd, anchor_time):
    if anchor_time:
        t = parse_time(anchor_time)
        log(f"  anchor set by hand: {fmt_time(t)}")
        return TimeMap([(0.0, t)]), {"anchors": [], "slopes": [], "warnings": []}
    review = read_json(review_path(movie))
    if (review and review.get("anchors") and review["anchors"][0]["text"] == cuts["anchors"][0]["text"]
            and abs((review.get("runtime") or 0) - audio.duration) < 0.05):
        pts = [(a["time"] - review["anchors"][0]["time"], a["time"]) for a in review["anchors"]]
        log("  same copy that was analyzed: using its anchor times directly")
        return TimeMap(pts), {"anchors": review["anchors"], "slopes": [], "warnings": []}
    log("  locating anchor lines in this copy ...")
    lang = cuts.get("language")

    def words_in(a, b):
        return transcribe_span(audio, eng.asr, wd / "asr", a, b, language=lang)

    tm, rep = resolve(cuts, words_in, audio.duration)
    for a in rep["anchors"]:
        found = "NOT FOUND" if a["found"] is None else f"{fmt_time(a['found'])} (match {a['score']:.2f})"
        log(f"    {a['text'][:50]!r}: {found}")
    return tm, rep


def _check_output(path, src_info, replace_audio):
    info = media.probe(path)
    a_src = len(media.audio_streams(src_info))
    a_out = len(media.audio_streams(info))
    v_src = sum(1 for s in src_info["streams"] if s["codec_type"] == "video")
    v_out = sum(1 for s in info["streams"] if s["codec_type"] == "video")
    if v_out != v_src or a_out < 1:
        raise RuntimeError(f"output check failed: video {v_out}/{v_src}, audio {a_out}")
    d_src = float(src_info["format"].get("duration") or 0)
    d_out = float(info["format"].get("duration") or 0)
    if d_src and abs(d_out - d_src) > 2.0:
        raise RuntimeError(f"output duration {d_out:.1f}s differs from source {d_src:.1f}s")
    clean = [s for s in media.audio_streams(info) if s["clean"]]
    if not clean:
        raise RuntimeError("output has no clean track")
    log(f"  output ok: {v_out} video, {a_out} audio (clean = track {clean[-1]['order']}), "
        f"was {a_src} audio")


def _summarize(reports):
    out = {"ranges": len(reports)}
    for r in reports:
        out[r["status"]] = out.get(r["status"], 0) + 1
    heard = sum(1 for r in reports if r.get("heard_after"))
    if heard:
        out["still_heard"] = heard
    return out


# ------------------------------------------------------------------ batch
def find_videos(root):
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    for p in files:
        if p.suffix.lower() not in media.VIDEO_EXTS or p.name.startswith("."):
            continue
        if any(part.startswith(".") for part in p.relative_to(root.parent if root.is_file() else root).parts[:-1]):
            continue
        if re.search(r"(?i)[-_. ](sample|trailer)\.", p.name):
            continue
        yield p


def batch(root, eng, out_dir=None, in_place=False, force=False, dry_run=False, analyze_opts=None,
          apply_opts=None):
    if not in_place and not out_dir:
        raise SystemExit("batch needs --in-place or --output-dir")
    results = []
    vids = list(find_videos(root))
    log(f"{len(vids)} video files under {root}")
    for i, v in enumerate(vids, 1):
        log(f"\n[{i}/{len(vids)}] {v}")
        try:
            streams = media.audio_streams(media.probe(v))
            if any(s["clean"] for s in streams) and not force:
                log("  already has a clean track - skipping (use --force to redo)")
                results.append((v, "skipped"))
                continue
            if dry_run:
                results.append((v, "would process"))
                continue
            if not cuts_path(v).exists():
                analyze(v, eng, **(analyze_opts or {}))
            out = None if in_place else Path(out_dir) / v.relative_to(Path(root) if Path(root).is_dir() else v.parent)
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
            rep = apply(v, cuts_path(v), eng, output=out, in_place=in_place, **(apply_opts or {}))
            results.append((v, "ok " + json.dumps(rep["summary"])))
        except (Exception, SystemExit) as e:
            log(f"  FAILED: {e}")
            results.append((v, f"failed: {e}"))
    log("\nSummary:")
    for v, r in results:
        log(f"  {r:40.40}  {v.name}")
    return results


# ------------------------------------------------------------------ doctor
def doctor(download=False, model="small.en", demucs_model="htdemucs"):
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {name:14} {detail}")

    print(f"moviefilter {__version__}, Python {sys.version.split()[0]}")
    for tool in ("ffmpeg", "ffprobe"):
        p = shutil.which(tool)
        ver = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.split("\n")[0] if p else ""
        line(tool, bool(p), ver or "not found on PATH")
    for mod in ("numpy", "scipy", "faster_whisper", "torch", "demucs", "requests"):
        try:
            m = __import__(mod)
            line(mod, True, getattr(m, "__version__", ""))
        except Exception as e:
            line(mod, mod == "requests", f"missing ({e.__class__.__name__}) - pip install -r requirements.txt")
    try:
        import torch
        line("gpu", True, "CUDA available - fast" if torch.cuda.is_available() else
             "none - CPU works, Demucs is ~1-3 min per 10 min of cussing-dense audio")
    except Exception:
        pass
    if download:
        try:
            Transcriber(model).model
            line("whisper model", True, model)
        except Exception as e:
            line("whisper model", False, str(e)[:120])
        try:
            from .separate import DemucsSeparator
            DemucsSeparator(demucs_model)._load()
            line("demucs model", True, demucs_model)
        except Exception as e:
            line("demucs model", False, str(e)[:120])
    print("all good" if ok else "fix the items marked !!")
    return ok
