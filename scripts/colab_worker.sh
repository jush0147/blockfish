#!/usr/bin/env bash
set -uo pipefail

BRANCH="rescue/independent-validation-20261116"
POLL_SECS="${POLL_SECS:-15}"
STATE_FILE=".colab_last_cmd_blob"

export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p results models

echo "================================================================"
echo " Blockfish Colab Auto-Polling Worker"
echo " Branch : $BRANCH"
echo " Cores  : $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo " Poll   : every ${POLL_SECS}s for changes in scripts/colab_cmd.sh"
echo "================================================================"

while true; do
  if ! git fetch origin "$BRANCH" --quiet 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] git fetch failed, retrying in ${POLL_SECS}s..."
    sleep "$POLL_SECS"
    continue
  fi

  REMOTE_BLOB="$(git rev-parse "origin/${BRANCH}:scripts/colab_cmd.sh" 2>/dev/null || echo "")"
  LAST_BLOB="$(cat "$STATE_FILE" 2>/dev/null || echo "")"

  if [ -n "$REMOTE_BLOB" ] && [ "$REMOTE_BLOB" != "$LAST_BLOB" ]; then
    echo ""
    echo "================================================================"
    echo "[$(date -u +%H:%M:%S)] NEW scripts/colab_cmd.sh detected ($REMOTE_BLOB)!"
    echo "================================================================"

    git checkout "$BRANCH" --quiet
    git reset --hard "origin/${BRANCH}" --quiet

    SHORT_SHA="$(git rev-parse --short HEAD)"
    echo "Running scripts/colab_cmd.sh at commit $SHORT_SHA..."
    set +e
    bash scripts/colab_cmd.sh 2>&1 | tee results/colab_latest.log
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    echo ""
    echo "[$(date -u +%H:%M:%S)] scripts/colab_cmd.sh finished (exit_code=$EXIT_CODE)"
    echo "$REMOTE_BLOB" > "$STATE_FILE"

    # Pull latest remote commits before committing results (in case code was pushed during run)
    git add -f results/colab_latest.log
    git add -f models/*.txt 2>/dev/null || true

    if ! git diff --cached --quiet; then
      git commit -m "colab: experiment results (${SHORT_SHA}, exit=${EXIT_CODE}) [skip ci]"
      git pull --rebase origin "$BRANCH" || true
      git push origin "$BRANCH" || echo "[WARN] git push failed — check GH_TOKEN permissions!"
      echo "=== DONE & PUSHED! Waiting for next scripts/colab_cmd.sh update... ==="
    else
      echo "=== DONE (no file changes to push). Waiting for next update... ==="
    fi
  fi

  sleep "$POLL_SECS"
done
