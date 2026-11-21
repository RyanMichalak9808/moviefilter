# Changelog

All notable changes to `moviefilter.py`.

## [0.4.0] - 2026-11-21
Rewritten as a package (`python -m moviefilter`, or `pip install .` for a `moviefilter`
command). `python moviefilter.py ...` still works.

### Accuracy
- Word list moved to `moviefilter/data/wordlist.json` with **levels** (`strong`, `slurs`,
  `mild`, `religious`), **phrases** ("son of a bitch", "oh my god"), stretched spellings
  ("fuuuck"), words Whisper censors itself ("f***"), and fuzzy matching for low-confidence
  near-misses. `--wordlist`, `--also`, `--allow` to customize. Starter lists for es/fr/de.
- Whisper is prompted so it doesn't skip swear words. Beam search and a more sensitive VAD
  are used.
- **Voice-only re-listen**: unclear passages are re-transcribed on the isolated voice
  (`--stem-transcribe refine`, default), or the whole film (`full`).
- Transcription runs in overlapping 10-minute chunks, so words on a boundary aren't lost.
- **Boundary snapping**: cuts snap to where the voice actually starts and stops instead of
  trusting Whisper's timestamps. This catches late or early timestamps and spares neighbouring words.
- **Verification**: every cleaned spot is re-separated (leftover voice measured) and
  re-transcribed. Failures get a second pass, then a 30 dB duck as last resort, and every case
  is reported.
- Flags possible names ("Dick"), low-confidence and fuzzy hits for review.
- Voice-pass hits that conflict with a word the main pass was sure of become **maybe**
  suggestions (not removed until accepted). Where the main pass was unsure, the voice pass may
  correct it. Tuned on the repeated test runs, which turned up both a mishearing
  ("Chicago" -> "shit") and a real word only the voice pass caught.
- Verification no longer uses the swear-word prompt, and leftover-voice measurement outranks
  Whisper. A spot whose voice measures gone is marked **check** instead of being ducked when
  Whisper "hears" the word from context. Also found by the repeated runs.

### Review
- New `review` command: table of detections with context, `--drop`, `--restore`, `--add`,
  `--levels`, `--flagged`, and `--clips DIR` (WAV per detection + `review.html` to listen).
- Edits are kept in `MOVIE.review.json` and survive re-running `analyze`.

### Audio
- **Surround preserved**: 5.1/7.1 stays 5.1/7.1. Front pair and center are cleaned separately
  (`--surrounds` for rears). LFE untouched. Native sample rate kept.
- **Sync fix**: encoder priming (AAC 1024 / AC-3 256 samples) is measured and compensated,
  so the clean track is sample-aligned with the original. 0.3.0 was ~21 ms late.
- Audio outside the cleaned spots is passed through unchanged, so loudness matches the original.
- `--codec aac|ac3|eac3|flac`, `--bitrate`. Defaults: AAC for stereo, AC-3 for surround.

### Workflow
- **Resumable**: transcription chunks and cleaned windows are cached. An interrupted run picks up
  where it stopped.
- Streams the audio instead of writing a full-length WAV (old version needed 1.3 GB+ per film).
- `--audio-track N`. Otherwise the default track is used and commentary tracks are skipped.
- Disk-space check before starting (`--no-space-check` to skip).
- Progress with ETA for every long step.
- `batch FOLDER --in-place|--output-dir` for whole libraries. Skips trailers, samples and
  already-cleaned files, and keeps going on errors with a summary at the end.
- `--in-place` keeps the original in a hidden `.moviefilter-backup/` folder.
- `doctor [--download]` checks FFmpeg, packages, GPU and models.

### Sharing / anchors
- Cuts file v2: **several anchors** (start, middle, end), runtime and levels. Still contains
  no words. v0.2/0.3 files still load.
- Handles a different intro length, **PAL speed-up** and other frame-rate differences, and edit
  differences between anchors (piecewise mapping, with a warning).
- Re-applying on the same copy skips the anchor search.

### Plex
- `--make-default`, `--plex-url/--plex-token/--plex-section` (or `PLEX_URL`/`PLEX_TOKEN`)
  to trigger a rescan of just that folder. Library section auto-detected from the path.
- Re-running replaces the existing clean track instead of adding another one.
- MP4 output names the track via `handler_name` too.

### Project
- README, `requirements.txt`, `pyproject.toml`, MIT `LICENSE`.
- Tests: 90 fast tests (fake engines + real FFmpeg), plus a `--slow` suite with real Whisper
  and Demucs on synthetic films with known ground truth (stereo, longer intro, PAL, 5.1).
- Measured on the final code (5 renderings with different voices, speeds, loudness and music;
  4 copies each, CPU, `small.en` + `htdemucs`): 25/25 slow tests pass. Every scripted cuss line
  was found with no false removals. Across 180 cleaned spots the original voice is 25-82 dB down
  (median 37 dB), the background stays within -2.1..+0.8 dB of the original, and there is 0 samples
  of sync error on all 20 outputs. Synthetic TTS speech, not real films: real-world accuracy
  depends mostly on the Whisper model.

## [0.3.0] - 2026-11-21
### Added
- Plex support: `apply` now adds the clean mix as an **extra audio track** inside the
  movie file instead of replacing the original audio. Plex lists it in the player's
  audio menu.
- `--track-title` to set the name Plex shows (default: "Clean (no profanity)").
- `--replace-audio` to keep the old behavior (swap the audio, single track).
- `count_audio_streams()` helper (ffprobe) so the new track is tagged and flagged correctly.

### Changed
- The original audio track stays the default; the clean track is not flagged default.
- All other streams (video, subtitles, extra audio) are copied untouched with `-map 0 -c copy`.

### Notes
- Plex has no support for a separate audio file next to the movie, so the track
  lives in the file. `.mkv` is recommended.
- Tested the mux step on a synthetic file only.

## [0.2.0] - 2026-11-21
### Added
- Vocal removal in place of muting: Demucs separates voice from music/effects in a
  window around each cuss word, and only the voice is dropped, so there is no silent gap.
- 15 ms crossfade at splice points to avoid clicks.
- `plan_windows()` groups nearby cuss words into shared Demucs windows (far faster
  than separating the whole film).
- `--demucs-model` and `--tmp` options.

### Removed
- Scene flagging (`scenes`, `add-skip`), skip ranges and the video re-encode path.
  The tool now only handles cuss words.
- Plain volume-mute approach.

### Notes
- Output audio is the first track downmixed to stereo 44.1 kHz.
- Crossfade and window-grouping logic tested on synthetic data; Whisper and
  Demucs were not run end to end.

## [0.1.0] - 2026-11-21
### Added
- Initial script: extract audio, split into 10-minute tracks, transcribe with
  faster-whisper (word timestamps), detect profanity (strong list, `--mild` for more).
- Anchor system: timestamps stored relative to a spoken anchor line (default: first
  8 words) in `<movie>.cuts.json`, so the file works on other rips of the same cut.
- Parental-guide scraping (IMDb, Kids-In-Mind, any URL) with optional Claude-assisted
  scene placement; manual `add-skip`.
- `apply` with ffmpeg mutes and scene skips.
