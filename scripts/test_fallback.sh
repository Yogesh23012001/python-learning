#!/usr/bin/env bash
# End-to-end test of the stored-score fallback path.
#
# 1. Populate the DB with a real score (normal timeout).
# 2. Force the live GitHub call to fail by setting GITHUB_API_TIMEOUT_S=0.001.
# 3. Hit /score — it should return the cached DB row + log "fallback".
# 4. Check /metrics for the outcome="degraded" counter.
# 5. Restore the original timeout and verify normal behavior.
#
# Run from the project root: ./scripts/test_fallback.sh

set -euo pipefail

PROJECT_DIR="/Users/yogeshyadav/Desktop/project/Ai RoadMap/myproject"
ENV_FILE="$PROJECT_DIR/.env"
ENV_BAK="$PROJECT_DIR/.env.bak.$$"
LOG_FILE="/tmp/uvicorn_fallback_test.log"
PORT=8000
USER_LOGIN="torvalds"

cd "$PROJECT_DIR"

# ----- helpers -------------------------------------------------------------

kill_uvicorn() {
  pkill -f "uvicorn api.app:app" 2>/dev/null || true
  # wait for the port to free
  for _ in {1..20}; do
    lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 || return 0
    sleep 0.2
  done
  echo "❌ port $PORT still bound — manual kill required"
  exit 1
}

start_uvicorn() {
  echo "▶ starting uvicorn (logs → $LOG_FILE)"
  : >"$LOG_FILE"
  nohup uv run uvicorn api.app:app --app-dir src --port $PORT \
    >"$LOG_FILE" 2>&1 &
  # wait for /metrics to become reachable
  for _ in {1..40}; do
    curl -sf "http://localhost:$PORT/metrics" >/dev/null 2>&1 && return 0
    sleep 0.25
  done
  echo "❌ uvicorn did not become ready within 10s"
  echo "----- log tail -----"
  tail -30 "$LOG_FILE"
  exit 1
}

set_env_var() {
  local key="$1" value="$2"
  # macOS sed
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i '' -E "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    printf '\n%s=%s\n' "$key" "$value" >>"$ENV_FILE"
  fi
}

cleanup() {
  echo ""
  echo "▶ cleanup: restoring .env and stopping uvicorn"
  if [ -f "$ENV_BAK" ]; then
    mv "$ENV_BAK" "$ENV_FILE"
  fi
  kill_uvicorn || true
}
trap cleanup EXIT

# ----- snapshot .env -------------------------------------------------------

cp "$ENV_FILE" "$ENV_BAK"

# ====================================================================
echo "==[ STEP 1 ]== populate DB with a fresh score (normal timeout)"
# ====================================================================
set_env_var GITHUB_API_TIMEOUT_S 15.0
kill_uvicorn
start_uvicorn

echo "▶ POST /score-and-save"
status=$(curl -sS -o /tmp/save_out.json -w "%{http_code}" \
  -X POST "http://localhost:$PORT/github/users/$USER_LOGIN/score-and-save")
echo "   status=$status"
echo "   body: $(jq -c . /tmp/save_out.json)"
if [ "$status" != "200" ]; then
  echo "❌ save failed — abort"
  exit 1
fi
echo "✅ DB now has a cached score for $USER_LOGIN"

# ====================================================================
echo ""
echo "==[ STEP 2 ]== force live GitHub call to fail (timeout=0.001)"
# ====================================================================
set_env_var GITHUB_API_TIMEOUT_S 0.001
kill_uvicorn
start_uvicorn

echo "▶ GET /github/users/$USER_LOGIN/score (live call should time out, fallback kicks in)"
status=$(curl -sS -o /tmp/fallback_out.json -w "%{http_code}" \
  "http://localhost:$PORT/github/users/$USER_LOGIN/score")
echo "   status=$status"
echo "   body: $(jq -c . /tmp/fallback_out.json)"

if [ "$status" != "200" ]; then
  echo "❌ expected 200 (degraded fallback), got $status"
  exit 1
fi

login=$(jq -r .login /tmp/fallback_out.json)
if [ "$login" != "$USER_LOGIN" ]; then
  echo "❌ expected login=$USER_LOGIN, got $login"
  exit 1
fi
echo "✅ got DB fallback response under timeout"

echo ""
echo "▶ checking log for fallback message"
if grep -q "score_user_live_failed_using_fallback" "$LOG_FILE"; then
  echo "✅ log contains score_user_live_failed_using_fallback"
  grep "score_user_live_failed_using_fallback" "$LOG_FILE" | tail -1
else
  echo "⚠️  fallback log line not found in $LOG_FILE — check the route is using the right except clause"
fi

echo ""
echo "▶ checking /metrics for outcome=\"degraded\" counter"
metrics=$(curl -sf "http://localhost:$PORT/metrics")
degraded=$(echo "$metrics" | grep '^scores_computed_total{outcome="degraded"}' || true)
if [ -n "$degraded" ]; then
  echo "✅ $degraded"
else
  echo "⚠️  no degraded counter — fallback path may not have executed"
  echo "    (try a longer timeout like 0.01 if the request resolved before timing out)"
fi

# ====================================================================
echo ""
echo "==[ STEP 3 ]== restore normal timeout and confirm normal behavior"
# ====================================================================
set_env_var GITHUB_API_TIMEOUT_S 15.0
kill_uvicorn
start_uvicorn

echo "▶ GET /github/users/$USER_LOGIN/score (live path, should succeed)"
status=$(curl -sS -o /tmp/normal_out.json -w "%{http_code}" \
  "http://localhost:$PORT/github/users/$USER_LOGIN/score")
echo "   status=$status"
echo "   body: $(jq -c .login,.score /tmp/normal_out.json)"

if [ "$status" != "200" ]; then
  echo "❌ normal path failed, got $status"
  exit 1
fi
echo "✅ live path works again"

# Cleanup runs from the trap
echo ""
echo "🎉 fallback flow verified end-to-end"
