# Changelog

All notable changes to `moviefilter.py`.

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
