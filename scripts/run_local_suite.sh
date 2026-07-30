#!/usr/bin/env bash
set -euo pipefail

DATASET_KIND="${1:-negative}"
SAMPLE_SIZE="${2:-100}"
PRESET="${3:-core}"
shift $(( $# >= 3 ? 3 : $# )) || true

case "$PRESET" in
  tiny)
    MODEL_KEYS=(
      ollama_qwen3_0_6b
      ollama_qwen3_1_7b
      ollama_deepseek_r1_1_5b
    )
    ;;
  core)
    MODEL_KEYS=(
      ollama_qwen3_4b
      ollama_qwen3_8b
      ollama_deepseek_llm_7b_chat
      ollama_deepseek_r1_7b
      ollama_deepseek_r1_8b
      ollama_glm4_9b
      ollama_mistral_7b
    )
    ;;
  chinese)
    MODEL_KEYS=(
      ollama_qwen3_1_7b
      ollama_qwen3_4b
      ollama_qwen3_8b
      ollama_deepseek_r1_1_5b
      ollama_deepseek_llm_7b_chat
      ollama_deepseek_r1_7b
      ollama_deepseek_r1_8b
      ollama_glm4_9b
    )
    ;;
  *)
    echo "用法: $0 [negative|positive|mixed] [样本数] {tiny|core|chinese} [其他 run_zero_shot 参数]" >&2
    exit 2
    ;;
esac

python scripts/check_local_models.py --models "${MODEL_KEYS[@]}"
python scripts/run_zero_shot.py \
  --dataset-kind "$DATASET_KIND" \
  --sample-size "$SAMPLE_SIZE" \
  --seed 42 \
  --workers 1 \
  --models "${MODEL_KEYS[@]}" \
  "$@"
