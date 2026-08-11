#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"
PROFILE="canonical"

if [[ "${1:-}" == "--hardware-profile" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--hardware-profile requires a profile name" >&2
    exit 2
  fi
  PROFILE="$2"
  shift 2
fi

PROFILE_ARGS=()
if [[ "$PROFILE" != "canonical" ]]; then
  PROFILE_OUTPUT="$(
    "$ROOT/.venv/bin/python" scripts/hardware_profile.py sft-args "$PROFILE"
  )"
  mapfile -t PROFILE_ARGS <<< "$PROFILE_OUTPUT"
fi

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa \
  "${PROFILE_ARGS[@]}" \
  "$@"

exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
