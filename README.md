# moviefilter

Remove cuss words from movies you own **without leaving silent gaps**. Only the voice is
removed for the length of the word. The music, sound effects and ambience keep playing
underneath. The cleaned audio is added as an **extra audio track** ("Clean (no profanity)")
that you pick from Plex's audio menu. The original track is untouched.

## How it works

1. **Find.** The film's audio is transcribed with [Whisper](https://github.com/SYSTRAN/faster-whisper)
   (word timestamps), then matched against a word list with levels, phrases ("son of a bitch"),
   stretched spellings ("fuuuck"), Whisper's own censoring ("f***") and near-misses.
   Passages Whisper wasn't sure about get a second listen on the **isolated voice** (music removed).
2. **Review** (optional). List everything found, listen to clips, drop false hits, add misses.
3. **Clean.** Around each word, [Demucs](https://github.com/facebookresearch/demucs) splits the
   audio into voice and background. The cut is **snapped to where the voice actually starts and
   stops**, and that stretch is replaced by the background with a 15 ms crossfade.
4. **Verify.** Each cleaned spot is separated again to measure leftover voice and re-transcribed
   to confirm the word is gone. If it isn't, it gets a second pass, and as a last resort it is
   ducked by 30 dB. Every case is reported.
5. **Add the track.** The clean track is encoded (AAC for stereo, AC-3 for 5.1/7.1),
   sample-aligned with the original (encoder delay compensated), and muxed into the file with
   all video, audio, subtitle and chapter streams copied as-is.

Surround is kept as surround: the front and center channels are cleaned separately
(`--surrounds` also cleans the rear channels). The LFE channel is never touched.

## Install

Needs Python 3.9+ and [FFmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
git clone https://github.com/RyanMichalak9808/moviefilter
cd moviefilter
pip install -r requirements.txt
python -m moviefilter doctor --download
```

`doctor --download` checks everything and fetches the models (~600 MB, once).
A GPU (CUDA) makes it much faster but is optional.

## Use

```bash
# 1. find the cuss words  ->  Movie.cuts.json (shareable) + Movie.review.json (private)
python -m moviefilter analyze "Movie (2010).mkv"

# 2. (optional) check what was found
python -m moviefilter review "Movie (2010).mkv"
python -m moviefilter review "Movie (2010).mkv" --clips review/   # WAV clips + review.html to listen
python -m moviefilter review "Movie (2010).mkv" --drop 7 12       # false hits
python -m moviefilter review "Movie (2010).mkv" --add 1:02:03.4-1:02:04.1   # a word it missed

# 3. add the clean track (original kept in .moviefilter-backup/ next to the movie)
python -m moviefilter apply "Movie (2010).mkv" --in-place
#    or write a new file instead
python -m moviefilter apply "Movie (2010).mkv" -o "Movie (2010).clean.mkv"
```

Everything resumes. If a run is interrupted, run the same command again and finished work
is reused from the cache (`~/.cache/moviefilter`, or set `--workdir` / `$MOVIEFILTER_CACHE`).

### What gets removed

`--levels` picks the categories (default `strong,slurs`):

| level | examples |
|---|---|
| `strong` | f-word, s-word, b-word, c-word, a-hole, bastard, goddamn, "son of a b…" |
| `slurs` | racial and homophobic slurs |
| `mild` | damn, hell, ass, crap, piss, bloody |
| `religious` | "oh my God", Jesus, Christ |

```bash
python -m moviefilter analyze movie.mkv --levels all
python -m moviefilter analyze movie.mkv --levels strong,slurs,mild --also twerp --allow "dick"
```

Everything is detected regardless of level, so `review --levels ...` can change your mind later
without re-analyzing. To customize the list, copy `moviefilter/data/wordlist.json`, edit it, and
pass `--wordlist my.json`. English is complete. Spanish, French and German have starter lists
(`--language es|fr|de|auto`).

### Plex

```bash
python -m moviefilter batch /media/Movies --in-place \
    --plex-url http://localhost:32400 --plex-token YOUR_TOKEN
```

* `batch` processes every video in the folder (skipping trailers/samples and files that already
  have a clean track) and asks Plex to rescan each folder, so the new track appears right away.
  Without the Plex options, run "Scan Library Files" yourself.
* In the player, open the audio menu and choose **Clean (no profanity)**. Add `--make-default`
  to make it the default track instead.
* Use `.mkv`. MP4 works, but track names are less reliable there.
* The clean track is re-encoded (AAC/AC-3). If you want it lossless, add `--codec flac`
  (bigger file: about 1 GB per hour for 5.1).
* Running again on the same movie *replaces* the clean track. It never stacks duplicates.

### Sharing a cuts file

`Movie.cuts.json` contains **no words**, only levels and times. Times are relative to a few
spoken "anchor" lines (start, middle, end of the film). Someone with their own copy runs:

```bash
python -m moviefilter apply "Their Copy.mkv" "Movie.cuts.json" --in-place
```

The anchors are found in their copy and every time is re-mapped. That handles different
intro/logo lengths and PAL speed-up (25 fps copies play about 4% faster). If the copies differ
between anchors (a different cut), it warns and interpolates. For a different edit (theatrical
vs. extended), run `analyze` on that copy instead.

## Accuracy tips

* **Bigger Whisper model = fewer misses**: `--model medium.en` or `--model large-v3` (with a GPU, or
  `distil-large-v3` for speed). The default `small.en` is a CPU-friendly middle ground.
* `--stem-transcribe full` transcribes the isolated voice for the whole film as well. It's the most
  thorough option, but slow on CPU.
* `review --flagged` shows only uncertain detections (low confidence, fuzzy match, a possible
  name like "Dick", or found only on the voice pass).
* Words found **only** by the voice-only re-listen, where the main pass was sure it heard a
  different word, show as **maybe**. They are *not* removed
  unless you accept them with `review --restore N`. That keeps a mishearing ("maps" → "shit")
  from cutting a clean line.
* After `apply`, spots are listed for you to listen to if they were **ducked** (the voice couldn't
  be separated) or marked **check** (the voice measured gone, but a re-transcription still claimed
  to hear the word, usually Whisper guessing from context).

## Limits

* Other voices talking *at the same moment* as the cuss word are removed for that fraction of a second too.
* Separation is very good but not perfect. On shouted or heavily processed lines a faint trace can
  remain. That's what verification and the ducking fallback are for.
* Whisper can still miss a mumbled word. Use `review --add` for those.
* CPU speed: roughly 10 s per cuss word for cleaning plus ~1x real-time to transcribe the film with
  `small.en` on a 4-core machine. A GPU is 5-20x faster.

## Tests

```bash
pip install pytest soundfile
pytest                      # ~90 fast tests: fake engines + real FFmpeg, ~40 s
pytest --slow               # real Whisper + Demucs on synthetic movies with known ground truth
MF_SEED=3 pytest --slow     # different voices/speeds/music
```

The slow suite needs `espeak-ng`. It builds test films from TTS dialogue and synthetic music, so
it can measure exactly how much voice was removed and how much background survived. It runs them
through a longer intro, a PAL speed-up and a 5.1 mix.

## License

MIT
