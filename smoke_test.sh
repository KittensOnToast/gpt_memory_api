#!/usr/bin/env bash
# path: smoke_test.sh
set -euo pipefail

# --- config / env ---
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

API_URL="${API_URL:-http://localhost:8001}"
API_KEY="${API_ACCESS_KEY:-${API_KEY:-}}"
USER_ID="${USER_ID:-u123}"
ROLE="${ROLE:-ops}"
QUERY="${QUERY:-deployment guide}"
TOP_K="${TOP_K:-3}"

if [ -z "${API_KEY}" ]; then
  echo "❌ Missing API key. Set API_ACCESS_KEY (or API_KEY) in .env or env." >&2
  exit 1
fi

pp() { python -m json.tool 2>/dev/null || cat; }

# --- helpers ---
http_json() {
  # $1: method, $2: url, $3: json-body (optional)
  local m="$1" u="$2" b="${3:-}"
  if [ -n "$b" ]; then
    curl -sS -w '\n%{http_code}' -X "$m" "$u" \
      -H "x-api-key: $API_KEY" -H "content-type: application/json" \
      -d "$b"
  else
    curl -sS -w '\n%{http_code}' -X "$m" "$u" \
      -H "x-api-key: $API_KEY"
  fi
}

show_or_die() {
  # expects combined output from http_json
  local res="$1"
  local code body
  code="$(echo "$res" | tail -n1)"
  body="$(echo "$res" | sed '$d')"
  echo "$body" | pp || true
  if [ "$code" != "200" ]; then
    echo "❌ HTTP $code" >&2
    exit 1
  fi
  printf '%s' "$body"
}

json_get() {
  # read JSON from stdin, extract a key (simple path)
  python - "$1" <<'PY'
import sys, json
data=json.loads(sys.stdin.read())
key=sys.argv[1]
# support one-level or nested a.b.c
cur=data
for part in key.split('.'):
    cur=cur[part]
print(cur)
PY
}

# --- wait for healthz ---
echo "▶ wait: /healthz at $API_URL"
for i in {1..20}; do
  set +e
  res=$(curl -s -o /dev/null -w '%{http_code}' -H "x-api-key: $API_KEY" "$API_URL/healthz")
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && [ "$res" = "200" ]; then break; fi
  sleep 0.5
done

# --- 1) healthz ---
echo "▶ 1) /healthz"
show_or_die "$(http_json GET "$API_URL/healthz")" >/dev/null

# --- 2) save ---
echo "▶ 2) /memory/save"
SAVE_BODY=$(printf '{"user_id":"%s","role":"%s","content":"The deploy playbook is at /runbook.","tags":["runbook","deploy"]}' "$USER_ID" "$ROLE")
SAVE_JSON="$(show_or_die "$(http_json POST "$API_URL/memory/save" "$SAVE_BODY")")"
MEM_ID="$(printf '%s' "$SAVE_JSON" | json_get memory_id)"
[ -n "$MEM_ID" ] || { echo "❌ no memory_id"; exit 1; }

# --- 3) lexical ---
echo "▶ 3) /memory/query (lexical)"
LEX_BODY=$(printf '{"user_id":"%s","role":"%s","query":"%s","top_k":%d,"mode":"lexical"}' "$USER_ID" "$ROLE" "runbook" "$TOP_K")
show_or_die "$(http_json POST "$API_URL/memory/query" "$LEX_BODY")" >/dev/null

# --- 4) reindex ---
echo "▶ 4) /embed/reindex"
REINDEX_JSON="$(show_or_die "$(http_json POST "$API_URL/embed/reindex" '{}')")"
JOB_ID="$(printf '%s' "$REINDEX_JSON" | json_get job_id)"
[ -n "$JOB_ID" ] || { echo "❌ no job_id"; exit 1; }
echo "job_id=$JOB_ID"

# --- 5) poll ---
echo "▶ 5) /jobs/$JOB_ID"
while :; do
  JOB_JSON="$(show_or_die "$(http_json GET "$API_URL/jobs/$JOB_ID")")"
  STATUS="$(printf '%s' "$JOB_JSON" | json_get status)"
  if [ "$STATUS" = "done" ]; then break; fi
  if [ "$STATUS" = "error" ]; then
    ERR="$(printf '%s' "$JOB_JSON" | json_get error || true)"
    echo "❌ Job error: $ERR" >&2
    exit 1
  fi
  sleep 1
done

# --- 6) hybrid ---
echo "▶ 6) /memory/query (hybrid)"
HYB_BODY=$(printf '{"user_id":"%s","role":"%s","query":"%s","top_k":%d,"mode":"hybrid"}' "$USER_ID" "$ROLE" "$QUERY" "$TOP_K")
show_or_die "$(http_json POST "$API_URL/memory/query" "$HYB_BODY")" >/dev/null

# --- 7) context ---
echo "▶ 7) /context/build"
CTX_BODY=$(printf '{"user_id":"%s","role":"%s","query":"%s","top_k":%d,"mode":"general"}' "$USER_ID" "$ROLE" "$QUERY" "$TOP_K")
show_or_die "$(http_json POST "$API_URL/context/build" "$CTX_BODY")" >/dev/null

echo "✅ Smoke test complete."
