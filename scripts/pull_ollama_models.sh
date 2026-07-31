#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "qwen3.5:9b"
  "glm4:9b"
  "deepseek-r1:8b"
  "llama3.1:latest"
  "mistral:latest"
)

command -v ollama >/dev/null 2>&1 || {
  echo "未找到 ollama 命令。请先安装并启动 Ollama。" >&2
  exit 2
}

for model in "${MODELS[@]}"; do
  echo "================================================================"
  echo "Pulling $model"
  ollama pull "$model"
done
