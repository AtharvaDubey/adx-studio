# ADx Studio

Make lyric videos on your Mac: pick a song, paste the lyrics, auto-sync every word to the vocal, choose an animated style, adjust the look and render a finished music video. Everything runs locally.

## Features

- **New video in two steps**: song, lyrics, format (16:9, 9:16, 1:1, 4:5, 21:9) and a style template.
- **Auto sync**: separates the vocal (Spleeter) and aligns each lyric word with a multilingual speech model (Omnilingual ASR, ONNX). Handles romanised Hindi/English lyrics. Models (~330 MB) download on first use.
- **Tap sync and timeline editing** for fine-tuning: drag words, resize, nudge, ripple, undo.
- **7 templates**: Ink & Ember, Kinetic Slam, Neon Night, Clean Karaoke, Typewriter, Cinematic, Social Captions. Colours, fonts, size, position, logo intro/outro, title card, particles, audio bars, beat pulse, background image.
- **Colour grading**: exposure, contrast, highlights, shadows, saturation, warmth, tint, fade, vignette, presets.
- **Fast preview engine**: low-res frames in tens of milliseconds, live while playing.
- **Rendering**: Draft / HD / 4K60, split across parallel workers, Apple VideoToolbox encoder when available.
- Opens custom code-generated film projects (zip with `analysis/timeline.json`) for lyric, timing and colour edits.

## Install (Mac app)

```bash
bash mac/build_app.sh                  # creates dist/ADx Studio.app
```
Drag it into Applications. It is unsigned, so the first time run
`xattr -dr com.apple.quarantine "/Applications/ADx Studio.app"`.
The first launch creates a private Python environment and installs the components.

## Run from source

```bash
pip3 install skia-python uharfbuzz opencv-python-headless numpy scipy pillow soundfile imageio-ffmpeg sherpa-onnx onnxruntime
python3 server.py        # http://127.0.0.1:8765
```

Projects are stored in `~/Documents/ADx Studio/Projects` when launched from the app (or `./projects` from source).

## Layout

| Path | What |
|---|---|
| `server.py` | local web server + project/render/sync API |
| `static/` | the app UI, logo, template previews |
| `engine/` | template renderer copied into each new project |
| `studio_worker.py` | persistent preview engine |
| `studio_render.py` | parallel segment renderer |
| `studio_analyze.py` | tempo, beats, loudness |
| `studio_autosync.py` | vocal separation + forced alignment |
| `look.py` | colour grade |
| `mac/` | app launcher, icon, build script |

Fonts are under the SIL Open Font License. The ADx logo belongs to ADx.
