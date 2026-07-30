# 本地小模型运行指南（Ollama）

本工程通过 OpenAI-compatible `/v1/chat/completions` 调用本地模型。对 macOS，推荐先使用 Ollama；运行器和指标代码无需修改。

## 1. 预置模型

| model key | Ollama tag | 参数规模 | 类型 |
|---|---|---:|---|
| `ollama_qwen3_0_6b` | `qwen3:0.6b` | 0.6B | Qwen3，thinking 关闭 |
| `ollama_qwen3_1_7b` | `qwen3:1.7b` | 1.7B | Qwen3，thinking 关闭 |
| `ollama_qwen3_4b` | `qwen3:4b` | 4B | Qwen3，thinking 关闭 |
| `ollama_qwen3_8b` | `qwen3:8b` | 8B | Qwen3，thinking 关闭 |
| `ollama_deepseek_llm_7b_chat` | `deepseek-llm:7b-chat` | 7B | 普通 DeepSeek Chat |
| `ollama_deepseek_r1_1_5b` | `deepseek-r1:1.5b` | 1.5B | R1 蒸馏，thinking 关闭 |
| `ollama_deepseek_r1_7b` | `deepseek-r1:7b` | 7B | R1 蒸馏，thinking 关闭 |
| `ollama_deepseek_r1_8b` | `deepseek-r1:8b` | 8B | R1 0528，thinking 关闭 |
| `ollama_glm4_9b` | `glm4:9b` | 9B | GLM-4 Chat |
| `ollama_mistral_7b` | `mistral:7b` | 7B | Mistral 7B Instruct v0.3 |

另外提供：

- `ollama_qwen3_4b_thinking`
- `ollama_deepseek_r1_7b_thinking`

这两个只用于 thinking ablation，不建议和非 thinking 主结果混在同一列比较。

## 2. 下载模型

小规模自检：

```bash
bash scripts/pull_ollama_models.sh tiny
```

论文核心本地模型：

```bash
bash scripts/pull_ollama_models.sh core
```

全部预置模型：

```bash
bash scripts/pull_ollama_models.sh all
```

也可以单独下载：

```bash
ollama pull qwen3:4b
ollama pull deepseek-llm:7b-chat
ollama pull deepseek-r1:7b
ollama pull glm4:9b
ollama pull mistral:7b
```

## 3. 检查服务和保存模型快照

```bash
python scripts/check_local_models.py
```

脚本会检查模型是否已安装，并把模型名称、digest、量化信息和修改时间保存到：

```text
runs/ollama_model_snapshot.json
```

论文实验必须保存这个快照。Ollama 的同一 tag 可能在未来更新，仅记录 `qwen3:4b` 不足以完全复现。

## 4. 先跑 10 条冒烟测试

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 10 \
  --seed 42 \
  --workers 1 \
  --models ollama_qwen3_1_7b
```

## 5. 跑一组模型

Negative 100 条，小模型：

```bash
bash scripts/run_local_suite.sh negative 100 tiny
```

Negative 300 条，核心模型：

```bash
bash scripts/run_local_suite.sh negative 300 core
```

中文模型为主的套件：

```bash
bash scripts/run_local_suite.sh negative 300 chinese
```

Mixed 300 条：

```bash
bash scripts/run_local_suite.sh mixed 300 core
```

`run_local_suite.sh` 默认 `seed=42`、`workers=1`，全部模型共享同一个样本清单。

## 6. 指定同一个冻结样本清单

第一轮运行结束后，找到：

```text
runs/zero_shot/<suite>/sample_manifest.jsonl
```

后续模型必须复用它：

```bash
python scripts/run_zero_shot.py \
  --manifest runs/zero_shot/<suite>/sample_manifest.jsonl \
  --workers 1 \
  --models \
    ollama_qwen3_4b \
    ollama_deepseek_llm_7b_chat \
    ollama_deepseek_r1_7b \
    ollama_glm4_9b \
    ollama_mistral_7b
```

## 7. 为什么保存 raw_output 和 output

推理模型可能通过服务端单独返回 thinking，也可能把推理包装成：

```text
<think>...</think>
最终句子
```

工程会：

- 将原始文本保存为 `raw_output`；
- 只移除完整闭合的 `<think>...</think>` 区块；
- 将实际参与过改率计算的文本保存为 `output`；
- 不删除解释、前缀、引号或代码块，这些仍算未遵守输出要求。

这样既避免把隐藏推理误算为改写，又不会人为美化模型结果。

## 8. 推荐的论文模型组合

第一轮不必把每个参数档位都跑满。建议：

1. Qwen3：1.7B、4B、8B，观察规模趋势；
2. DeepSeek-LLM-7B-Chat：普通 DeepSeek 指令模型；
3. DeepSeek-R1：1.5B、7B 或 8B，观察推理蒸馏是否影响过改；
4. GLM-4-9B：中文模型家族对照；
5. Mistral-7B：非中文专门模型对照。

主结果统一使用：

- 同一 prompt；
- 同一 `sample_manifest.jsonl`；
- temperature 0；
- seed 42；
- thinking 关闭；
- 同一 Ollama 量化口径，或至少完整报告 digest 和 quantization。

thinking 版本应作为单独消融实验。
