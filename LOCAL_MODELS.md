# 本地小模型运行指南（Ollama，v2.3 clean）

本项目通过 Ollama 的 OpenAI-compatible `/v1/chat/completions` 接口运行本地模型。

## 当前五个小模型

| model key | Ollama tag | 参数规模 |
|---|---|---:|
| `qwen3_5_9b_ollama` | `qwen3.5:9b` | 9B |
| `glm4_9b_ollama` | `glm4:9b` | 9B |
| `deepseek_r1_8b_ollama` | `deepseek-r1:8b` | 8B |
| `llama3_1_8b_ollama` | `llama3.1:latest` | 8B |
| `mistral_7b_ollama` | `mistral:latest` | 7B |

## 1. 检查 Ollama 与模型

```bash
ollama list
python scripts/check_local_models.py \
  --models \
    qwen3_5_9b_ollama \
    glm4_9b_ollama \
    deepseek_r1_8b_ollama \
    llama3_1_8b_ollama \
    mistral_7b_ollama
```

## 2. 先跑 100 条

```bash
./run_all_small_models.sh 100
```

五个模型在同一个 suite 中运行，共享同一份 `sample_manifest.jsonl`。

## 3. 再跑 Negative 全量 717 条

```bash
./run_all_small_models.sh full
```

## 4. 单模型冒烟测试

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 5 \
  --seed 42 \
  --workers 1 \
  --max-retries 0 \
  --models qwen3_5_9b_ollama
```

## 5. 结果目录

```text
runs/zero_shot/<suite>/
├── sample_manifest.jsonl
├── suite_config.json
├── model_comparison.csv
├── model_comparison.json
├── qwen3_5_9b_ollama/
├── glm4_9b_ollama/
├── deepseek_r1_8b_ollama/
├── llama3_1_8b_ollama/
└── mistral_7b_ollama/
```

论文比较时保留 `sample_manifest.jsonl`、`suite_config.json` 和 `runs/ollama_model_snapshot.json`。
