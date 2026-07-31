# GIFR Zero-shot Baseline

本目录实现中文性别包容审查的多模型 zero-shot 直接改写基线。

核心实验假设：模型收到同一条 zero-shot 指令后，若输出与原句不同，则视为触发了 `EDIT`。

- Golden Negative：gold action 为 `KEEP`，任何实质变化计为 over-edit。
- Golden Positive：gold action 为 `EDIT`，原样输出计为 no-edit；但“输出变化”不等于改写正确。
- Mixed：报告 KEEP/EDIT 触发层面的代理分类指标。

## 当前数据

当前使用 v2.3 main-only clean 工作簿；`数据集` sheet 中的记录均为 `处置=主集`：

- `data/source_jsonl/negative_main.jsonl`：717 条
- `data/source_jsonl/positive_main.jsonl`：871 条
- `data/source_jsonl/all_main.jsonl`：1588 条

正式运行链路只读取 JSON/JSONL，Excel 只作为人工维护源。

## 1. 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 中设置所需 provider 的 Key：

```bash
DEEPSEEK_API_KEY=你的DeepSeekKey
DASHSCOPE_API_KEY=你的百炼Key
ZHIPU_API_KEY=你的智谱Key
```

Qwen/GLM 的详细配置与 401 排查见 [`API_MODELS.md`](API_MODELS.md)。

## 2. 先做 dry-run

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 100 \
  --models qwen3_5_9b_ollama \
  --dry-run
```

Dry-run 会原样回显输入。Negative 上应得到 0% over-edit，用于验证数据、抽样和指标代码。


## 2A. v2.3 clean 小模型重测

先跑五个小模型各 100 条（同一 suite、同一 manifest）：

```bash
./run_all_small_models.sh 100
```

确认后跑 Golden Negative 全量 717 条：

```bash
./run_all_small_models.sh full
```

详细命令见 [`command.md`](command.md)。

## 3. 跑 DeepSeek

Negative 100 条：

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 100 \
  --seed 42 \
  --models deepseek_v4_flash
```

同一批 300 条同时比较 Flash 与 Pro：

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 300 \
  --seed 42 \
  --models deepseek_v4_flash deepseek_v4_pro
```

Mixed 300 条，默认 Negative/Positive 各 150 条：

```bash
python scripts/run_zero_shot.py \
  --dataset-kind mixed \
  --sample-size 300 \
  --seed 42 \
  --models deepseek_v4_flash deepseek_v4_pro
```

全量必须显式指定：

```bash
python scripts/run_zero_shot.py --dataset-kind negative --full --models deepseek_v4_flash
```

## 3A. 跑 Qwen 与 GLM

先用 3 条检查 Key、区域和模型配置：

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 3 \
  --workers 1 \
  --models qwen3_7_plus glm5_2_zhipu
```

比较时必须复用 DeepSeek 300 条实验生成的 `sample_manifest.jsonl`：

```bash
python scripts/run_zero_shot.py \
  --manifest runs/zero_shot/<deepseek-suite>/sample_manifest.jsonl \
  --workers 2 \
  --requests-per-second 1 \
  --models qwen3_7_plus glm5_2_zhipu
```

Qwen Flash/Plus/Max 规模档位：

```bash
python scripts/run_zero_shot.py \
  --manifest runs/zero_shot/<deepseek-suite>/sample_manifest.jsonl \
  --models qwen3_7_flash qwen3_7_plus qwen3_7_max
```

## 4. 公平比较多个模型

一次命令中的所有模型共享同一份 `sample_manifest.jsonl`。输出结构：

```text
runs/zero_shot/<suite>/
├── sample_manifest.jsonl
├── suite_config.json
├── model_comparison.csv
├── model_comparison.json
├── deepseek_v4_flash/
│   ├── results.jsonl
│   ├── results.csv
│   ├── metrics.json
│   └── summary.txt
└── deepseek_v4_pro/
    └── ...
```

以后增加模型时，可冻结并复用同一份样本：

```bash
python scripts/run_zero_shot.py \
  --manifest runs/zero_shot/<suite>/sample_manifest.jsonl \
  --models 新模型key
```

## 5. 接入其他 API 或本地小模型

运行器使用 OpenAI-compatible Chat Completions 接口。DeepSeek、Qwen、GLM、OpenAI，
以及通过 vLLM、Ollama 或 LM Studio 暴露的本地模型，都只需在
`configs/models/zero_shot_models.json` 中增加配置，不改实验代码：

```json
{
  "local_model": {
    "provider": "openai_compatible",
    "base_url": "http://localhost:8000/v1",
    "api_key_env": "",
    "model": "填写服务端暴露的模型名",
    "temperature": 0.0,
    "max_tokens": 512,
    "size_label": "8B",
    "parameter_count": "8B"
  }
}
```

然后：

```bash
python scripts/run_zero_shot.py --dataset-kind negative --sample-size 300 --models local_model
```

## 6. 数据抽样规则

- 默认只保留 `处置=主集`。
- 默认按 `L1类别` 比例分层抽样。
- `--seed` 固定后，样本完全可复现。
- `mixed` 指定样本量时默认 1:1，可用 `--negative-ratio` 调整。
- `--split test` 可只跑指定切分。
- `--exclude-controversial` 可排除争议样本。
- 每次运行都会保存实际样本清单，后续模型必须复用该清单进行公平比较。

## 7. 比较口径

- `strict_changed`：只忽略首尾空白，其他字符变化全部算修改。
- `normalized_changed`：额外忽略 Unicode 全半角和空白格式差异。
- Negative 主指标：`strict_over_edit_rate` 与 `content_over_edit_rate`。
- Positive 只报告 `edit_trigger_rate`，不把它误称为改写成功率。
- Mixed 的 accuracy/F1 是编辑触发代理指标，最终论文仍需独立评估去偏成功与语义保持。

## 8. 从 Excel 重新导出 JSONL

```bash
python scripts/export_main_jsonl.py \
  --negative data/source_excel/GoldenNegative_v2.3_main_only_clean.xlsx \
  --positive data/source_excel/GoldenPositive_v2.3_main_only_clean.xlsx \
  --output-dir data/source_jsonl
```

该脚本只用于数据准备；正式 baseline 不依赖 Excel。
