#!/bin/zsh
# Open the research ROM in the visible desktop mGBA app. Load data/rogue-profile.lua
# in Tools > Scripting after it opens, then run `rogue-rl visual` in another terminal.
set -euo pipefail
root_dir=${0:A:h:h}
rom="$root_dir/data/rogue-research.gba"
if [[ ! -f "$rom" ]]; then
  print -u2 "Missing research ROM: $rom"
  exit 2
fi
open -na mGBA --args -3 "$rom"
