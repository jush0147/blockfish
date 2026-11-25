#!/usr/bin/env bash
set -uo pipefail

BRANCH="rescue/independent-validation-20261116"
POLL_SECS="${POLL_SECS:-15}"
LIVE_SYNC_SECS="${LIVE_SYNC_SECS:-120}"
STATE_FILE=".colab_last_cmd_blob"

export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p results models

echo "================================================================"
echo " Blockfish Colab Auto-Polling Worker (with Live Log Sync)"
echo " Branch : $BRANCH"
echo " Cores  : $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo " Poll   : every ${POLL_SECS}s | Live log push: every ${LIVE_SYNC_SECS}s"
echo "================================================================"

IDLE_TICKS=0
while true; do
  if ! git fetch origin "$BRANCH" --quiet 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] git fetch failed, retrying in ${POLL_SECS}s..."
    sleep "$POLL_SECS"
    continue
  fi

  REMOTE_BLOB="$(git rev-parse "origin/${BRANCH}:scripts/colab_cmd.sh" 2>/dev/null || echo "")"
  LAST_BLOB="$(cat "$STATE_FILE" 2>/dev/null || echo "")"

  if [ -n "$REMOTE_BLOB" ] && [ "$REMOTE_BLOB" != "$LAST_BLOB" ]; then
    IDLE_TICKS=0
    echo ""
    echo "================================================================"
    echo "[$(date -u +%H:%M:%S)] NEW scripts/colab_cmd.sh detected ($REMOTE_BLOB)!"
    echo "================================================================"

    git checkout "$BRANCH" --quiet
    git reset --hard "origin/${BRANCH}" --quiet

    SHORT_SHA="$(git rev-parse --short HEAD)"
    : > results/colab_latest.log
    echo "RUNNING commit=${SHORT_SHA} started=$(date -u +%Y-%m-%dT%H:%M:%SZ) cores=$(nproc)" > results/colab_status.txt
    git add -f results/colab_status.txt
    if ! git diff --cached --quiet; then
      git commit -m "colab: started (${SHORT_SHA}) [skip ci]" --quiet
      git push origin "$BRANCH" --quiet || true
    fi

    # Background live-log sync every LIVE_SYNC_SECS so nobody ever has to babysit the Colab UI
    (
      while true; do
        sleep "$LIVE_SYNC_SECS"
        {
          echo "RUNNING commit=${SHORT_SHA} heartbeat=$(date -u +%Y-%m-%dT%H:%M:%SZ) cores=$(nproc)"
          echo "--- tail results/colab_latest.log ---"
          tail -n 15 results/colab_latest.log 2>/dev/null || true
        } > results/colab_status.txt
        git add -f results/colab_status.txt 2>/dev/null || true
        if ! git diff --cached --quiet 2>/dev/null; then
          git commit -m "colab: live status (${SHORT_SHA}) [skip ci]" --quiet 2>/dev/null || true
          git pull --rebase origin "$BRANCH" --quiet 2>/dev/null || true
          git push origin "$BRANCH" --quiet 2>/dev/null || true
        fi
      done
    ) &
    SYNC_PID=$!

    echo "Running scripts/colab_cmd.sh at commit $SHORT_SHA..."
    set +e
    bash scripts/colab_cmd.sh 2>&1 | tee results/colab_latest.log
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true

    echo ""
    echo "[$(date -u +%H:%M:%S)] scripts/colab_cmd.sh finished (exit_code=$EXIT_CODE)"
    echo "$REMOTE_BLOB" > "$STATE_FILE"
    echo "DONE commit=${SHORT_SHA} exit=${EXIT_CODE} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/colab_status.txt

    git add -f results/colab_latest.log results/colab_status.txt
    git add -f models/*.txt 2>/dev/null || true

    if ! git diff --cached --quiet; then
      git commit -m "colab: experiment results (${SHORT_SHA}, exit=${EXIT_CODE}) [skip ci]"
      git pull --rebase origin "$BRANCH" || true
      git push origin "$BRANCH" || echo "[WARN] git push failed — check GH_TOKEN permissions!"
      echo "=== DONE & PUSHED! Waiting for next scripts/colab_cmd.sh update... ==="
    else
      echo "=== DONE (no file changes to push). Waiting for next update... ==="
    fi
  else
    IDLE_TICKS=$((IDLE_TICKS + 1))
    if [ $((IDLE_TICKS % 20)) -eq 0 ]; then
      echo "[$(date -u +%H:%M:%S)] worker alive, waiting for next scripts/colab_cmd.sh..."
    fi
  fi

  sleep "$POLL_SECS"
done
