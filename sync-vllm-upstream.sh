#!/usr/bin/env bash
# Sync the sibling vllm-semantic-cache checkout's `main` branch to
# upstream/vllm-project's main -- WITHOUT a real merge.
#
# Why not a real merge: this fork's `main` and upstream/main share a common
# ancestor from Feb 2023, so `git merge upstream/main` produces real
# add/add conflicts in unrelated core files (gpu_worker.py, gpu_model_
# runner.py, flash_attn_interface.py, speculator.py, ...) that have nothing
# to do with this project. The fix: confirm local main has zero commits
# actually authored by us (it's just a stale mirror, not real work), then
# hard-reset the branch ref straight to upstream/main. No merge, no
# conflicts. Verified safe this way on 2026-07-29 -- see project memory /
# conversation history if you want the full diagnosis.
#
# This only touches the LOCAL main branch. It does not push to origin and
# does not touch any feature branch (e.g. register-cache-policy-api).
#
# Usage:
#   ./sync-vllm-upstream.sh                 # repo at ../vllm-semantic-cache
#   VLLM_REPO=/path/to/vllm-semantic-cache ./sync-vllm-upstream.sh

set -euo pipefail

VLLM_REPO="${VLLM_REPO:-../vllm-semantic-cache}"
AUTHOR="${AUTHOR:-Philip Pesic}"

if [ ! -d "$VLLM_REPO/.git" ]; then
  echo "Not a git repo: $VLLM_REPO (set VLLM_REPO=/path/to/vllm-semantic-cache)"
  exit 1
fi

cd "$VLLM_REPO"
echo "Repo: $(pwd)"

if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "No 'upstream' remote configured. Add it first:"
  echo "  git remote add upstream https://github.com/vllm-project/vllm.git"
  exit 1
fi

orig_branch=$(git branch --show-current)

echo "Fetching upstream..."
git fetch upstream

echo "Safety check: any commits on main authored by '$AUTHOR' not in upstream/main?"
unique_commits=$(git log main --not upstream/main --author="$AUTHOR" --oneline)
if [ -n "$unique_commits" ]; then
  echo "STOP: found commits on main that upstream/main doesn't have:"
  echo "$unique_commits"
  echo "These would be silently discarded by a hard reset -- investigate"
  echo "before re-running (e.g. cherry-pick them onto a feature branch first)."
  exit 1
fi
echo "OK: none found -- safe to reset."

if [ -n "$(git status --porcelain -- ':!.serena')" ]; then
  echo "STOP: working tree has uncommitted changes outside .serena/. Commit,"
  echo "stash, or clean those up first:"
  git status --short
  exit 1
fi

git checkout main
git reset --hard upstream/main

echo
echo "main is now at: $(git log -1 --oneline)"
echo "(local only -- origin/main and any feature branches are untouched)"

if [ "$orig_branch" != "main" ] && [ -n "$orig_branch" ]; then
  git checkout "$orig_branch"
  echo "Switched back to $orig_branch"
fi

echo
echo "main jumped forward a large number of commits -- if you're about to"
echo "run anything GPU-side against it, rebuild the editable install first:"
echo "  cd $VLLM_REPO && VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto"
