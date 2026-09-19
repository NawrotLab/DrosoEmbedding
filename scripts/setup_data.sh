#!/bin/bash
# One-time data setup: extract the Data repository's packaged frame archives
# into the directory layout the code expects, and write a matching .env.
#
# Usage (from the code repository root):
#     bash scripts/setup_data.sh /path/to/DrosoEmbedding_WBCI
#
# Safe to re-run: archives whose recording directory already exists are
# skipped, and an existing .env is never overwritten.
set -euo pipefail

if [ $# -ne 1 ] || [ ! -d "$1/data/preprocessed_frames" ]; then
    echo "Usage: bash scripts/setup_data.sh /path/to/DrosoEmbedding_WBCI" >&2
    exit 1
fi
DATA_REPO="$(cd "$1" && pwd)"
CODE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="$DATA_REPO/data/preprocessed_frames"

# All under data/preprocessed_frames/ (the extracted directories are
# already git-ignored by the Data repository):
#   intact.tars/<rec>.tar               -> meanZ_allTs/<rec>/
#   ko_static_<neuropil>.tars/<rec>.tar -> meanZ_allTs_KO_static_<neuropil>/<rec>/
extract_all() {  # <dir of .tar files> <target dir>
    mkdir -p "$2"
    local n=0
    for t in "$1"/*.tar; do
        [ -s "$t" ] || { echo "  missing content: $t (run 'git annex get' first)" >&2; exit 1; }
        [ -d "$2/$(basename "$t" .tar)" ] || { tar -xf "$t" -C "$2"; n=$((n + 1)); }
    done
    echo "  $(basename "$1") -> $2 ($n newly extracted)"
}

echo "Extracting frame archives in $FRAMES"
extract_all "$FRAMES/intact.tars" "$FRAMES/meanZ_allTs"
for d in "$FRAMES"/ko_static_*.tars; do
    # KO archives are optional (only Fig S4 needs them) -- skip if not fetched
    first="$(ls "$d" 2>/dev/null | head -n 1)"
    [ -n "$first" ] && [ -s "$d/$first" ] || { echo "  $(basename "$d"): not fetched, skipping"; continue; }
    neuropil="$(basename "$d" .tars)"; neuropil="${neuropil#ko_static_}"
    extract_all "$d" "$FRAMES/meanZ_allTs_KO_static_$neuropil"
done

if [ -e "$CODE_REPO/.env" ]; then
    echo ".env already exists -- left untouched (DROSO_DATA_REPO should be $DATA_REPO)"
else
    sed -e "s|^DROSO_ROOT=.*|DROSO_ROOT=$CODE_REPO|" \
        -e "s|^DROSO_DATA_REPO=.*|DROSO_DATA_REPO=$DATA_REPO|" \
        "$CODE_REPO/.env.example" > "$CODE_REPO/.env"
    echo "Wrote $CODE_REPO/.env"
fi
