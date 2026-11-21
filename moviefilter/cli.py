"""Command-line interface.

  moviefilter analyze MOVIE            transcribe + detect -> MOVIE.cuts.json (shareable) + MOVIE.review.json
  moviefilter review  MOVIE            list detections; --drop/--restore/--add; --clips DIR to listen
  moviefilter apply   MOVIE CUTS ...   add a "Clean" audio track (voice removed, background kept)
  moviefilter batch   FOLDER ...       analyze + apply every video in a folder (e.g. a Plex library)
  moviefilter doctor                   check ffmpeg, Python packages, GPU, models
"""
import argparse
import os
import sys

from . import __version__, util
from .wordlist import DEFAULT_LEVELS, LEVELS


def _levels(s):
    lv = [x.strip() for x in s.split(",") if x.strip()]
    if lv == ["all"]:
        return list(LEVELS)
    bad = set(lv) - set(LEVELS)
    if bad:
        raise argparse.ArgumentTypeError(f"unknown level(s) {', '.join(bad)}; choose from {', '.join(LEVELS)} or all")
    return lv


def _engine_args(p):
    g = p.add_argument_group("speech recognition / separation")
    g.add_argument("--model", help="Whisper model (default small.en, or small for --language auto). "
                                   "Bigger = more accurate: medium.en, large-v3, distil-large-v3")
    g.add_argument("--language", default="en", help="spoken language code, or 'auto' (default en)")
    g.add_argument("--device", default="auto", help="auto | cpu | cuda")
    g.add_argument("--no-vad", action="store_true", help="don't skip non-speech (slower, can catch shouted lines)")
    g.add_argument("--demucs-model", default="htdemucs", help="htdemucs (default) or htdemucs_ft (better, 4x slower)")
    g.add_argument("--shifts", type=int, default=1, help="Demucs shifts: 2-5 = cleaner separation, slower")
    g.add_argument("--wordlist", help="custom word list JSON (copy moviefilter/data/wordlist.json)")
    g.add_argument("--also", action="append", default=[], metavar="WORD", help="extra word to remove (repeatable)")
    g.add_argument("--allow", action="append", default=[], metavar="REGEX", help="never treat this word as profanity")


def _analyze_args(p):
    g = p.add_argument_group("analysis")
    g.add_argument("--levels", type=_levels, default=list(DEFAULT_LEVELS),
                   help=f"what to remove: comma list of {', '.join(LEVELS)} or 'all' (default strong,slurs)")
    g.add_argument("--stem-transcribe", choices=["off", "refine", "full"], default="refine",
                   help="re-listen on the isolated voice: refine = unclear passages only (default), "
                        "full = whole film (slow, most thorough)")
    g.add_argument("--refine-max", type=float, default=15, help="max minutes of audio to re-listen (default 15)")
    g.add_argument("--anchor-text", help="spoken line to use as the first anchor")
    g.add_argument("--anchor-time", help="when that line starts in your copy (skips searching)")


def _apply_args(p):
    g = p.add_argument_group("output")
    g.add_argument("--verify", choices=["off", "fast", "full"], default="full",
                   help="check each cleaned spot: fast = measure leftover voice, "
                        "full = also re-transcribe it (default)")
    g.add_argument("--surrounds", action="store_true", help="also clean surround channels (5.1/7.1)")
    g.add_argument("--codec", help="aac | ac3 | eac3 | flac (default aac for stereo, ac3 for surround)")
    g.add_argument("--bitrate", help="e.g. 256k, 640k")
    g.add_argument("--track-title", default="Clean (no profanity)", help="name shown in Plex's audio menu")
    g.add_argument("--make-default", action="store_true", help="make the clean track the default track")
    g.add_argument("--replace-audio", action="store_true", help="output only the clean track")
    g.add_argument("--audio-track", type=int, help="source audio track number (0 = first); default: the default track")
    g.add_argument("--levels", type=_levels, help="only apply these levels from the cuts file")
    g.add_argument("--no-backup", action="store_true", help="with --in-place: don't keep the original")
    g.add_argument("--plex-url", default=os.environ.get("PLEX_URL"), help="e.g. http://localhost:32400 (or env PLEX_URL)")
    g.add_argument("--plex-token", default=os.environ.get("PLEX_TOKEN"), help="X-Plex-Token (or env PLEX_TOKEN)")
    g.add_argument("--plex-section", help="library section id (auto-detected from the path if omitted)")


def build_parser():
    p = argparse.ArgumentParser(prog="moviefilter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"moviefilter {__version__}")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--workdir", help="cache/scratch folder (default ~/.cache/moviefilter or $MOVIEFILTER_CACHE)")
    p.add_argument("--no-space-check", action="store_true")
    sp = p.add_subparsers(dest="cmd", required=True)

    a = sp.add_parser("analyze", help="find the cuss words")
    a.add_argument("movie")
    a.add_argument("--audio-track", type=int, help="source audio track number (0 = first)")
    _analyze_args(a)
    _engine_args(a)

    r = sp.add_parser("review", help="check / edit what was found")
    r.add_argument("movie")
    r.add_argument("--drop", type=int, nargs="+", default=[], metavar="N", help="don't remove row N (false hit)")
    r.add_argument("--restore", type=int, nargs="+", default=[], metavar="N", help="undo a --drop")
    r.add_argument("--add", action="append", default=[], metavar="START-END",
                   help="remove a missed word, e.g. 1:02:03.4-1:02:04.1 (repeatable)")
    r.add_argument("--add-level", default="manual")
    r.add_argument("--levels", type=_levels, help="change which levels go into the cuts file")
    r.add_argument("--flagged", action="store_true", help="only show rows flagged for review")
    r.add_argument("--clips", metavar="DIR", help="export a WAV per row + review.html to listen")

    ap = sp.add_parser("apply", help="write the clean audio track")
    ap.add_argument("movie")
    ap.add_argument("cuts", nargs="?", help="cuts file (default: MOVIE.cuts.json)")
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("-o", "--output", help="output file (.mkv recommended)")
    out.add_argument("--in-place", action="store_true", help="replace the movie file (original kept in .moviefilter-backup/)")
    ap.add_argument("--anchor-time", help="set where the first anchor line starts in this copy")
    _apply_args(ap)
    _engine_args(ap)

    b = sp.add_parser("batch", help="process every video in a folder")
    b.add_argument("folder")
    bo = b.add_mutually_exclusive_group(required=True)
    bo.add_argument("--in-place", action="store_true")
    bo.add_argument("--output-dir")
    b.add_argument("--force", action="store_true", help="redo files that already have a clean track")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--analyze-levels", type=_levels, default=list(DEFAULT_LEVELS), dest="a_levels")
    b.add_argument("--stem-transcribe", choices=["off", "refine", "full"], default="refine")
    _apply_args(b)
    _engine_args(b)

    d = sp.add_parser("doctor", help="check the setup")
    d.add_argument("--download", action="store_true", help="also download/check the default models")
    return p


def _engines(a):
    from .pipeline import Engines
    return Engines(model=a.model, language=a.language, device=a.device, vad=not a.no_vad,
                   demucs_model=a.demucs_model, shifts=a.shifts, wordlist=a.wordlist,
                   extra_words=a.also, allow=a.allow)


def _apply_kw(a):
    plex = {"url": a.plex_url, "token": a.plex_token, "section": a.plex_section} if a.plex_url else None
    return dict(backup=not a.no_backup, levels=a.levels, audio_track=a.audio_track, verify=a.verify,
                surrounds=a.surrounds, codec=a.codec, bitrate=a.bitrate, track_title=a.track_title,
                make_default=a.make_default, replace_audio=a.replace_audio, workdir=a.workdir,
                space_check=not a.no_space_check, plex=plex)


def main(argv=None):
    a = build_parser().parse_args(argv)
    util.QUIET = a.quiet
    from . import pipeline as pl
    try:
        if a.cmd == "analyze":
            pl.analyze(a.movie, _engines(a), levels=a.levels, stem_mode=a.stem_transcribe,
                       refine_max=a.refine_max * 60, audio_track=a.audio_track, anchor_text=a.anchor_text,
                       anchor_time=a.anchor_time, workdir=a.workdir, space_check=not a.no_space_check)
            print(f"Next: moviefilter review \"{a.movie}\"   then   moviefilter apply \"{a.movie}\" --in-place")
        elif a.cmd == "review":
            pl.review_cmd(a.movie, drop=a.drop, restore=a.restore, add=a.add, add_level=a.add_level,
                          levels=a.levels, flagged_only=a.flagged, clips=a.clips, workdir=a.workdir)
        elif a.cmd == "apply":
            from .cuts import cuts_path
            pl.apply(a.movie, a.cuts or cuts_path(a.movie), _engines(a), output=a.output, in_place=a.in_place,
                     anchor_time=a.anchor_time, **_apply_kw(a))
        elif a.cmd == "batch":
            kw = _apply_kw(a)
            res = pl.batch(a.folder, _engines(a), out_dir=a.output_dir, in_place=a.in_place, force=a.force,
                           dry_run=a.dry_run,
                           analyze_opts={"levels": a.a_levels, "stem_mode": a.stem_transcribe,
                                         "workdir": a.workdir, "space_check": not a.no_space_check},
                           apply_opts=kw)
            if any(r.startswith("failed") for _, r in res):
                sys.exit(1)
        elif a.cmd == "doctor":
            sys.exit(0 if pl.doctor(download=a.download) else 1)
    except KeyboardInterrupt:
        print("\ninterrupted - rerun the same command to resume (finished work is cached)", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
