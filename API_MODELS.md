# Qwen 与 GLM API 运行指南

本版本预置以下 API 模型配置：

| model key | 实际模型 | 接口 | thinking |
|---|---|---|---|
| `qwen3_7_flash` | `qwen3.7-flash-2026-07-15` | DashScope | 关闭 |
| `qwen3_7_plus` | `qwen3.7-plus-2026-05-26` | DashScope | 关闭 |
| `qwen3_7_max` | `qwen3.7-max-2026-06-08` | DashScope | 关闭 |
| `glm5_2_zhipu` | `glm-5.2` | 智谱直连 | 关闭 |
| `glm5_2_dashscope` | `glm-5.2` | DashScope | 关闭 |

## 1. 配置密钥

```bash
cp .env.example .env
```

Qwen：

```bash
DASHSCOPE_API_KEY=你的百炼Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

GLM 智谱直连：

```bash
ZHIPU_API_KEY=你的智谱Key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

`DASHSCOPE_API_KEY` 必须与 `DASHSCOPE_BASE_URL` 所属区域一致。若使用新加坡或其他区域，修改 `DASHSCOPE_BASE_URL`，不要只替换 Key。

## 2. 先做 3 条冒烟测试

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 3 \
  --seed 42 \
  --workers 1 \
  --models qwen3_7_plus glm5_2_zhipu
```

## 3. 复用 DeepSeek 的 300 条样本

不要重新执行 `--sample-size 300`。找到 DeepSeek 那次运行目录中的：

```text
runs/zero_shot/<deepseek-suite>/sample_manifest.jsonl
```

然后运行：

```bash
python scripts/run_zero_shot.py \
  --manifest runs/zero_shot/<deepseek-suite>/sample_manifest.jsonl \
  --workers 2 \
  --requests-per-second 1 \
  --models qwen3_7_plus glm5_2_zhipu
```

若要看 Qwen 规模档位差异：

```bash
python scripts/run_zero_shot.py \
  --manifest runs/zero_shot/<deepseek-suite>/sample_manifest.jsonl \
  --workers 2 \
  --requests-per-second 1 \
  --models qwen3_7_flash qwen3_7_plus qwen3_7_max
```

## 4. 推荐第一轮模型组合

先跑：

```text
DeepSeek-V4-Pro（已有）
Qwen3.7-Plus
GLM-5.2
```

这是跨模型家族比较。确认现象稳定后，再补：

```text
Qwen3.7-Flash / Plus / Max
本地 Qwen 1.7B / 4B / 8B
本地 GLM-4-9B
```

## 5. 401 排查

Qwen 返回 401 时依次检查：

1. `.env` 中是否是 `DASHSCOPE_API_KEY`，不是 DeepSeek 或智谱 Key；
2. Key 与 `DASHSCOPE_BASE_URL` 是否属于同一区域；
3. 模型是否在该区域可用；
4. 修改 `.env` 后重新启动命令，或执行 `set -a; source .env; set +a`；
5. 不要在同一 shell 中保留旧的 `DASHSCOPE_API_KEY`。
