#!/usr/bin/env bash
set -euo pipefail

# Default base URL (can be overridden by --base-url)
BASE_URL="http://127.0.0.1:8080"

# Parse optional --base-url flag
if [[ "${1:-}" == "--base-url" && -n "${2:-}" ]]; then
  BASE_URL="$2"
  shift 2
fi

API_KEY_FILE=".apikey"

# Ensure .apikey exists and is not empty
if [[ ! -f "$API_KEY_FILE" ]]; then
  echo "❌ ERROR: .apikey file not found."
  exit 1
fi
API_KEY="$(cat "$API_KEY_FILE")"
if [[ -z "$API_KEY" ]]; then
  echo "❌ ERROR: .apikey is empty."
  exit 1
fi

# Quick auth check
AUTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" -H "x-api-key: $API_KEY" "$BASE_URL/healthz")
if [[ "$AUTH_CHECK" != "200" ]]; then
  echo "❌ ERROR: API key rejected by $BASE_URL (HTTP $AUTH_CHECK)."
  exit 1
fi

echo "✅ API key check passed. Running smoke tests against $BASE_URL..."

echo "=== 1) Health check (no auth) ==="
curl -s "$BASE_URL/healthz" | jq .

echo "=== 2) File upload (multipart) ==="
echo "test file content" > test1.txt
curl -s -X POST "$BASE_URL/files/upload?user_id=testuser" \
  -H "x-api-key: $API_KEY" \
  -F "file=@test1.txt" | tee upload_multipart.json | jq .

echo "=== 3) File upload (JSON base64) ==="
if base64 --help 2>&1 | grep -q -- " -w "; then
  B64_CONTENT=$(base64 -w 0 test1.txt)
else
  B64_CONTENT=$(base64 < test1.txt | tr -d '\n')
fi
curl -s -X POST "$BASE_URL/files/upload-json" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"testuser\",\"filename\":\"test2.txt\",\"content_base64\":\"$B64_CONTENT\"}" \
  | tee upload_json.json | jq .

echo "=== 4) File upload (URL - GitHub raw) ==="
curl -s -X POST "$BASE_URL/files/upload-url" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"testuser","url":"https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore","filename":"remote.txt"}' \
  | tee upload_url.json | jq .

echo "=== 5) List files ==="
curl -s "$BASE_URL/files/list?user_id=testuser" -H "x-api-key: $API_KEY" | jq .

echo "=== 6) Download file ==="
curl -s -O -J "$BASE_URL/files/download/test1.txt" -H "x-api-key: $API_KEY" && echo "Downloaded test1.txt ✅"

echo "=== 7) Delete file ==="
curl -s -X DELETE "$BASE_URL/files/delete/test1.txt?user_id=testuser" -H "x-api-key: $API_KEY" | jq .

echo "=== 8) Save memory ==="
MEM_ID=$(curl -s -X POST "$BASE_URL/memory/save" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"user_id":"testuser","role":"assistant","content":"This is a test memory"}' \
  | jq -r .memory_id)
echo "Memory ID: $MEM_ID"

echo "=== 9) Query memory ==="
curl -s -X POST "$BASE_URL/memory/query" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d "{\"user_id\":\"testuser\",\"role\":\"assistant\",\"query\":\"test\",\"top_k\":3}" \
  | jq .

echo "=== 10) Update memory ==="
curl -s -X POST "$BASE_URL/memory/update" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d "{\"memory_id\":\"$MEM_ID\",\"role\":\"assistant\",\"new_content\":\"Updated content\"}" \
  | jq .

echo "=== 11) Auto-query memory ==="
curl -s -X POST "$BASE_URL/memory/auto-query" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d "{\"user_id\":\"testuser\",\"role\":\"assistant\",\"query\":\"updated\",\"top_k\":2}" \
  | jq .

echo "=== 12) Feedback memory ==="
curl -s -X POST "$BASE_URL/memory/feedback" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d "{\"memory_id\":\"$MEM_ID\",\"role\":\"assistant\",\"user_id\":\"testuser\",\"feedback_type\":\"positive\",\"feedback_text\":\"Looks good\"}" \
  | jq .

echo "=== 13) Tag search memory ==="
curl -s -X POST "$BASE_URL/memory/tag-search" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"role":"assistant","tags":[],"top_k":3}' \
  | jq .

echo "=== 14) Self-review memory ==="
curl -s "$BASE_URL/memory/self-review" -H "x-api-key: $API_KEY" | jq .

echo "=== 15) Save goal ==="
curl -s -X POST "$BASE_URL/memory/goals" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"user_id":"testuser","content":"My first goal"}' \
  | jq .

echo "=== 16) List goals ==="
curl -s "$BASE_URL/memory/goals" -H "x-api-key: $API_KEY" | jq .

echo "=== 17) Build context ==="
curl -s -X POST "$BASE_URL/context/build" \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"user_id":"testuser","role":"assistant","query":"updated","top_k":3,"mode":"summarise"}' \
  | jq .

echo "=== Smoke tests completed successfully ==="

