#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-path> [extra key=value overrides...]"
  exit 1
fi

CONFIG_PATH="$1"
shift

cd "${REPO_ROOT}"
export PYTHONPATH=.

torchrun --nproc_per_node=1 dinov3/eval/segmentation/run.py \
  "config=${CONFIG_PATH}" \
  "$@"
