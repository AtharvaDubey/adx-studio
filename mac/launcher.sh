#!/bin/bash
# ADx Studio launcher: starts the local studio server (if it isn't running) and opens the app window.
APP="$(cd "$(dirname "$0")/../Resources/app" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export ADX_HOME="$HOME/Documents/ADx Studio"
mkdir -p "$ADX_HOME/Projects"
LOG="$ADX_HOME/adx-studio.log"
PORT=8765
URL="http://127.0.0.1:$PORT"

open_window() {
  if [ -d "/Applications/Google Chrome.app" ]; then
    open -na "Google Chrome" --args --app="$URL" --window-size=1500,920
  else
    open "$URL"
  fi
}

if curl -s -m 1 "$URL/api/ping" | grep -q "ADx Studio"; then open_window; exit 0; fi

SYSPY="$(command -v python3)"
if [ -z "$SYSPY" ]; then
  osascript -e 'display alert "ADx Studio needs Python 3" message "Install Python from python.org, then open ADx Studio again."'
  exit 1
fi

# ADx Studio keeps its own private Python environment so it never mixes with other installs.
VENV="$HOME/Library/Application Support/ADx Studio/venv"
PY="$VENV/bin/python3"
PKGS="skia-python uharfbuzz opencv-python-headless numpy scipy pillow soundfile imageio-ffmpeg"
check() { "$PY" -s -c "import numpy, scipy, cv2, skia, uharfbuzz, PIL, imageio_ffmpeg" >/dev/null 2>&1; }
if ! check; then
  osascript -e 'display notification "Setting up ADx Studio (first launch). This takes a few minutes." with title "ADx Studio"'
  rm -rf "$VENV"; mkdir -p "$(dirname "$VENV")"
  { "$SYSPY" -m venv "$VENV" && "$PY" -m pip install --upgrade pip && "$PY" -m pip install $PKGS; } >>"$LOG" 2>&1
  if ! check; then
    osascript -e "display alert \"ADx Studio could not set up its components\" message \"Details are in $LOG\""
    exit 1
  fi
fi
# auto-sync components (optional: the rest of the app works without them)
if ! "$PY" -s -c "import sherpa_onnx, onnxruntime" >/dev/null 2>&1; then
  osascript -e 'display notification "Installing the auto-sync components…" with title "ADx Studio"'
  "$PY" -m pip install sherpa-onnx onnxruntime >>"$LOG" 2>&1 || true
fi
export PYTHONNOUSERSITE=1

cd "$APP"
"$PY" server.py --port "$PORT" --no-browser >>"$LOG" 2>&1 &
SERVER=$!
for i in $(seq 1 60); do
  curl -s -m 1 "$URL/api/ping" >/dev/null && break
  if ! kill -0 $SERVER 2>/dev/null; then osascript -e "display alert \"ADx Studio failed to start\" message \"Details are in $LOG\""; exit 1; fi
  sleep 0.25
done
open_window
trap 'kill $SERVER 2>/dev/null' TERM INT
wait $SERVER
