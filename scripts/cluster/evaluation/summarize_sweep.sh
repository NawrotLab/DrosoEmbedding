#!/bin/bash
# Scans logs/evaluation/*.log for the "Total wall time for <run_id>: <s>s"
# line run_evaluation.py's main() logs on completion, and produces one
# summary: how many of the manifest's models finished, total/mean/min/max
# duration, and which ones (if any) never completed at all.
#
# Usage (from repo root, on the cluster, after the array job has run --
# safe to run again later too, e.g. after resubmitting stragglers):
#   bash scripts/cluster/evaluation/summarize_sweep.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MANIFEST="scripts/cluster/evaluation/eval_manifest.txt"
LOG_DIR="logs/evaluation"

if [ ! -f "$MANIFEST" ]; then
  echo "No manifest at $MANIFEST -- run build_manifest.sh first."
  exit 1
fi

TIMES_FILE=$(mktemp)
trap 'rm -f "$TIMES_FILE"' EXIT

# One "run_id seconds" pair per completed run, deduped to the latest entry
# per run_id (a run_id can appear more than once if it was resubmitted).
grep -h "Total wall time for" "$LOG_DIR"/*.log 2>/dev/null \
  | sed -E 's/.*Total wall time for ([^:]+): ([0-9.]+)s.*/\1 \2/' \
  | awk '{ t[$1] = $2 } END { for (r in t) print r, t[r] }' \
  > "$TIMES_FILE"

n_total=$(wc -l < "$MANIFEST")
n_done=$(wc -l < "$TIMES_FILE")

echo "=== Evaluation sweep summary ==="
echo "Manifest entries: $n_total"
echo "Completed (have a logged wall time): $n_done"
echo ""

if [ "$n_done" -gt 0 ]; then
  awk '
    { sum += $2; n++; if (min == "" || $2 < min) min = $2; if ($2 > max) max = $2 }
    END {
      printf "Duration (s): total=%.1f  mean=%.1f  min=%.1f  max=%.1f\n", sum, sum/n, min, max
      printf "Estimated wall time if run fully serially: %.1f min\n", sum/60
    }
  ' "$TIMES_FILE"
fi

echo ""
echo "--- Not yet completed (no logged wall time found) ---"
awk '{print $2}' "$MANIFEST" | sort > /tmp/_manifest_runids.txt
awk '{print $1}' "$TIMES_FILE" | sort > /tmp/_done_runids.txt
comm -23 /tmp/_manifest_runids.txt /tmp/_done_runids.txt || true
n_missing=$(comm -23 /tmp/_manifest_runids.txt /tmp/_done_runids.txt | wc -l)
rm -f /tmp/_manifest_runids.txt /tmp/_done_runids.txt
echo ""
echo "$n_missing not yet completed."
