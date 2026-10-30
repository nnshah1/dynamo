#!/bin/bash
# render.sh <name>... : d2 -> svg -> png (headless chrome + trim). Layout: dagre unless D2_LAYOUT set.
set -euo pipefail
cd "$(dirname "$0")"
for n in "$@"; do
  d2 --layout "${D2_LAYOUT:-dagre}" "$n.d2" "$n.svg" >/dev/null
  google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --window-size=2400,2400 --screenshot="$n.raw.png" "file://$PWD/$n.svg" >/dev/null 2>&1
  convert "$n.raw.png" -trim -bordercolor white -border 24 "$n.png" && rm -f "$n.raw.png"
  echo "$n: $(identify -format '%wx%h' "$n.png")"
done
