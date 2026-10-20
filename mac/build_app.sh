#!/bin/bash
# Builds "ADx Studio.app" in ./dist from this repository.
set -e
cd "$(dirname "$0")/.."
A="dist/ADx Studio.app/Contents"
rm -rf "dist/ADx Studio.app"; mkdir -p "$A/MacOS" "$A/Resources/app"
cp mac/Info.plist "$A/"; cp mac/AppIcon.icns "$A/Resources/"
cp mac/launcher.sh "$A/MacOS/ADxStudio"; chmod +x "$A/MacOS/ADxStudio"
cp -r server.py look.py studio_*.py deva_known.py engine fonts static "$A/Resources/app/"
echo "Built dist/ADx Studio.app"
