#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-core}"
case "$PRESET" in
  tiny)
    MODELS=(
      "qwen3:0.6b"
      "qwen3:1.7b"
      "deepseek-r1:1.5b"
    )
    ;;
  core)
    MODELS=(
      "qwen3:4b"
      "qwen3:8b"
      "deepseek-llm:7b-chat"
      "deepseek-r1:7b"
      "deepseek-r1:8b"
      "glm4:9b"
      "mistral:7b"
    )
    ;;
  all)
    MODELS=(
      "qwen3:0.6b"
      "qwen3:1.7b"
      "qwen3:4b"
      "qwen3:8b"
      "deepseek-r1:1.5b"
      "deepseek-llm:7b-chat"
      "deepseek-r1:7b"
      "deepseek-r1:8b"
      "glm4:9b"
      "mistral:7b"
    )
    ;;
  *)
    echo "用法: $0 {tiny|core|all}" >&2
    exit 2
    ;;
esac

command -v ollama >/dev/null 2>&1 || {
  echo "未找到 ollama 命令。请先安装并启动 Ollama。" >&2
  exit 2
}

for model in "${MODELS[@]}"; do
  echo "================================================================"
  echo "Pulling $model"
  ollama pull "$model"
done
