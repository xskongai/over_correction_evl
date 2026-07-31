#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./run_all_small_models.sh 100      # 先跑 100 条 pilot（推荐）
#   ./run_all_small_models.sh full     # 跑清理后的 717 条 Golden Negative 全量
#   ./run_all_small_models.sh 10 --dry-run

SIZE="${1:-100}"
shift || true

MODEL_KEYS=(
  qwen3_5_9b_ollama
  glm4_9b_ollama
  deepseek_r1_8b_ollama
  llama3_1_8b_ollama
  mistral_7b_ollama
)

if [[ "$SIZE" == "full" ]]; then
  SIZE_ARGS=(--full)
elif [[ "$SIZE" =~ ^[1-9][0-9]*$ ]]; then
  SIZE_ARGS=(--sample-size "$SIZE")
else
  echo "样本数必须是正整数或 full，例如：$0 100 / $0 full" >&2
  exit 2
fi

python scripts/run_zero_shot.py \
  --dataset-kind negative \
  "${SIZE_ARGS[@]}" \
  --seed 42 \
  --workers 1 \
  --max-retries 0 \
  --models "${MODEL_KEYS[@]}" \
  "$@"

echo
echo "模型运行结束，正在生成论文结果表……"
python scripts/summarize_run.py --require-complete
