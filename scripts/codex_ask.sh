#!/usr/bin/env bash
# Usage: codex_ask.sh new <prompt_file> <out_file> [effort] [model]
#        codex_ask.sh resume <session_id> <prompt_file> <out_file> [effort] [model]
set -euo pipefail
cd /home/myid/hj67104/consultation_saturation
MODE="$1"; shift
if [ "$MODE" = "new" ]; then
  PROMPT_FILE="$1"; OUT="$2"; EFFORT="${3:-xhigh}"; MODEL="${4:-gpt-5.4}"
  RAW="logs/codex_raw_$(basename "$OUT" .md).log"
  codex exec --skip-git-repo-check --model "$MODEL" -c model_reasoning_effort="$EFFORT" \
    --sandbox read-only --output-last-message "$OUT" - < "$PROMPT_FILE" > "$RAW" 2>&1 || true
else
  SID="$1"; PROMPT_FILE="$2"; OUT="$3"; EFFORT="${4:-xhigh}"; MODEL="${5:-gpt-5.4}"
  RAW="logs/codex_raw_$(basename "$OUT" .md).log"
  codex exec resume "$SID" --skip-git-repo-check --model "$MODEL" -c model_reasoning_effort="$EFFORT" \
    -c sandbox_mode="read-only" --output-last-message "$OUT" - < "$PROMPT_FILE" > "$RAW" 2>&1 || true
fi
grep -m1 -oE 'session id: [0-9a-f-]{36}' "$RAW" | sed 's/session id: /SESSION_ID=/' || echo "SESSION_ID=UNKNOWN"
if [ -s "$OUT" ]; then echo "OK bytes=$(wc -c < "$OUT")"; else echo "EMPTY_OUTPUT see $RAW"; fi
