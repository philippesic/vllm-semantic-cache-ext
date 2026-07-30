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
# Also picks a reset target that actually has a precompiled wheel for this
# machine's architecture: wheels.vllm.ai's x86_64 cu130 builds have been
# observed lagging aarch64 by dozens of commits (a whole day, once), and
# setup.py's VLLM_USE_PRECOMPILED path resolves the wheel commit via a live
# GitHub-API call to upstream's tip + merge-base against local HEAD (see
# setup.py get_base_commit_in_main_branch) -- so landing on a tip commit
# with no wheel for our arch makes `uv pip install -e .` fail later. We
# walk backward from upstream/main until we find a commit with a published
# wheel, so the eventual install just works. Override CUDA_VARIANT if this
# box isn't on cu130; set SKIP_WHEEL_CHECK=1 to fall back to old
# tip-of-main behavior (e.g. no network access to wheels.vllm.ai).
#
# Usage:
#   ./sync-vllm-upstream.sh                 # repo at ../vllm-semantic-cache
#   VLLM_REPO=/path/to/vllm-semantic-cache ./sync-vllm-upstream.sh
#   CUDA_VARIANT=cu128 ./sync-vllm-upstream.sh
#   SKIP_WHEEL_CHECK=1 ./sync-vllm-upstream.sh

set -euo pipefail

VLLM_REPO="${VLLM_REPO:-../vllm-semantic-cache}"
AUTHOR="${AUTHOR:-Philip Pesic}"
CUDA_VARIANT="${CUDA_VARIANT:-cu130}"
ARCH="$(uname -m)"
WHEEL_SEARCH_DEPTH="${WHEEL_SEARCH_DEPTH:-300}"

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

target="$(git rev-parse upstream/main)"

if [ "${SKIP_WHEEL_CHECK:-0}" != "1" ] && command -v curl >/dev/null 2>&1; then
  echo "Looking for the newest upstream/main commit with a $CUDA_VARIANT/$ARCH wheel..."
  found=""
  checked=0
  for sha in $(git rev-list upstream/main -n "$WHEEL_SEARCH_DEPTH"); do
    checked=$((checked + 1))
    has_arch=$(curl -s --max-time 10 \
      "https://wheels.vllm.ai/${sha}/${CUDA_VARIANT}/vllm/metadata.json" |
      python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
print('yes' if any('$ARCH' in f.get('platform_tag', '') for f in data) else 'no')
" 2>/dev/null || echo "no")
    if [ "$has_arch" = "yes" ]; then
      found="$sha"
      break
    fi
  done
  if [ -n "$found" ]; then
    target="$found"
    echo "Found: $target (checked $checked commit(s) back from tip)"
    if [ "$target" != "$(git rev-parse upstream/main)" ]; then
      echo "NOTE: this is behind upstream/main tip -- wheel publishing for"
      echo "$ARCH/$CUDA_VARIANT hasn't caught up yet. Re-run later for a newer pin."
    fi
  else
    echo "WARNING: no $CUDA_VARIANT/$ARCH wheel found in the last $checked" \
      "commits -- falling back to tip of upstream/main. The editable" \
      "install's VLLM_USE_PRECOMPILED step will likely fail; either widen" \
      "WHEEL_SEARCH_DEPTH or drop VLLM_USE_PRECOMPILED for a full source build."
  fi
else
  echo "Skipping wheel-availability check (SKIP_WHEEL_CHECK=1 or no curl) --" \
    "using tip of upstream/main."
fi

echo "Safety check: any commits on main authored by '$AUTHOR' not in the reset target?"
unique_commits=$(git log main --not "$target" --author="$AUTHOR" --oneline)
if [ -n "$unique_commits" ]; then
  echo "STOP: found commits on main that the reset target doesn't have:"
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
git reset --hard "$target"

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
