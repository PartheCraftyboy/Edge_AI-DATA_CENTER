#!/usr/bin/env bash
# Teach the classifier from the reader, overnight.
#
#   ./train.sh                 60 GNHK + 150 IAM pages already ingested, 3 rounds
#   ./train.sh --rounds 5      more rounds
#   ./train.sh --limit 200     stop after 200 new teacher calls, for a short run
#
# Everything it prints also goes to backend/runs_distil/train.log, so closing
# the terminal does not lose the run's record. Progress is visible three ways:
# this terminal, that log, and `palimtop live` in another terminal once the
# server is up.
set -euo pipefail

cd "$(dirname "$0")/backend"
PYTHON=.venv/bin/python
[ -x "$PYTHON" ] || { echo "no .venv here; see backend/requirements.txt" >&2; exit 1; }

mkdir -p runs_distil
LOG=runs_distil/train.log

# The teacher is a vision-language model served by Ollama, about 1.2 s a crop.
# Check it is actually up before spending an hour discovering it is not.
if ! curl -sf --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null; then
    echo "Ollama is not answering at ${OLLAMA_HOST:-http://localhost:11434}." >&2
    echo "Start it with:  ollama serve" >&2
    exit 1
fi

pages=0
for d in data/raw/gnhk/pages data/raw/iam/pages; do
    [ -d "$d" ] && pages=$((pages + $(find "$d" -name '*.png' | wc -l)))
done
echo "pages available to teach on: $pages"
if [ "$pages" -lt 20 ]; then
    echo "that is too few; ingest a corpus first:" >&2
    echo "    .venv/bin/python -m app.data.gnhk --limit 200" >&2
    echo "    .venv/bin/python -m app.data.iam  --limit 150" >&2
    exit 1
fi

IMAGES=()
[ -d data/raw/gnhk/pages ] && IMAGES+=(data/raw/gnhk/pages)
[ -d data/raw/iam/pages ] && IMAGES+=(data/raw/iam/pages)

echo "started $(date '+%F %T')" | tee -a "$LOG"
echo "log: backend/$LOG"
echo

# Every corpus in ONE pass, which is the whole point of this line.
#
# This used to loop, one `app.distil` per folder, and that silently threw
# work away: each run starts again from the synthetic checkpoint and writes
# the same runs_distil/<backbone>/best.ckpt, so the second corpus replaced
# the first rather than building on it. Training on GNHK and then on IAM
# shipped a model that had only ever seen IAM. The run said so plainly and
# it went unread for hours: the GNHK pass finished at 0.743 and the IAM pass
# then opened at 0.128, which is the untrained synthetic score.
#
# The corpus being discarded was the worse one to lose. GNHK is camera
# captured, with the shadow, skew and clutter the pipeline actually meets;
# IAM is flatbed scans. Having both exists to catch a model that has learned
# about scanners, and training them in sequence into one path defeats that.
#
# Votes are cached per page and box, so stopping and restarting costs
# nothing already bought: the teacher is never asked the same question twice.
echo "=== teaching on ${IMAGES[*]} ===" | tee -a "$LOG"
# -u, not stdbuf. Python does its own io buffering and block-buffers stdout
# whenever it is a pipe, which `tee` makes it; stdbuf's LD_PRELOAD does not
# reach that, so an earlier version printed nothing for minutes and read as
# a hang. PIPESTATUS, because a pipeline's exit code is tee's and tee always
# succeeds.
status=0
stdbuf -oL -eL "$PYTHON" -u -m app.distil --images "${IMAGES[@]}" "$@" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    status=1
    echo "!! training failed; see backend/$LOG" | tee -a "$LOG"
fi

echo | tee -a "$LOG"
echo "finished $(date '+%F %T')" | tee -a "$LOG"
echo "learning curve:  .venv/bin/python -m app.history --plot" | tee -a "$LOG"
"$PYTHON" -m app.history 2>/dev/null || true
exit "$status"
