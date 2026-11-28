#!/usr/bin/env bash
# WP-18: ingest each extra subreddit's submissions with the WP-16 ingester.
# The ingester always writes <out>/submissions.parquet, so each subreddit is ingested into
# its own staging directory and then moved next to (never over) the r/memes parquet.
set -euo pipefail
cd /home/linus/memes-revised
SRC=/home/linus/reddit_dl_wp18/reddit/subreddits25
DST=dataset_saves/reddit/submissions-2025-12
STAGE=experiments/WP-18/ingest_stage
for sub in dankmemes meirl comedyheaven whenthe 196 shitposting; do
  if [ -e "$DST/${sub}_submissions.parquet" ]; then echo "skip $sub"; continue; fi
  mkdir -p "$STAGE/$sub"
  uv run python experiments/wp16/ingest.py --archive "$SRC/${sub}_submissions.zst" --out "$STAGE/$sub"
  mv -n "$STAGE/$sub/submissions.parquet" "$DST/${sub}_submissions.parquet"
  mv -n "$STAGE/$sub/ingest_report.json" "$DST/${sub}_ingest_report.json"
done
sha256sum "$SRC"/*_submissions.zst "$DST"/*.parquet > experiments/WP-18/checksums.sha256
