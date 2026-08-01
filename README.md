# GIFR Zero-shot Baseline

本项目用于评估中文性别包容改写中的 over-edit / overcorrection。

核心判定：模型收到统一 zero-shot 指令后，若输出与原句不同，则视为触发 `EDIT`。

- Golden Negative：gold action 为 `KEEP`，实质变化计为 over-edit。
- Golden Positive：gold action 为 `EDIT`，原样输出计为 no-edit；但“发生变化”不等于改写正确。
- Mixed：报告 KEEP/EDIT 触发层面的代理指标。

## 当前数据

只使用清理后工作簿中 `处置=主集` 的数据：

- `data/source_jsonl/negative_main.jsonl`：717 条
- `data/source_jsonl/positive_main.jsonl`：871 条
- `data/source_jsonl/all_main.jsonl`：1588 条

正式运行只读取 JSONL；Excel 是人工维护源。

## 1. 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 中设置需要的 API Key：

```bash
DEEPSEEK_API_KEY=你的DeepSeekKey
DASHSCOPE_API_KEY=你的百炼Key
ZHIPU_API_KEY=你的智谱Key
```

本地模型还需要启动 Ollama。详见 `LOCAL_MODELS.md`；API 模型详见 `API_MODELS.md`。

## 2. 统一模型运行入口

现有的 `scripts/run_zero_shot.py` 本身已经支持任意多个模型，并保证它们共享同一份
`sample_manifest.jsonl`。在此基础上新增了一个很薄的通用入口：

```bash
./run_models.sh TARGET SIZE [其他 run_zero_shot.py 参数]
```

`TARGET` 可以是：

- 一个 model key；
- 逗号分隔的多个 model key；
- `small`、`large`、`all` 等模型组。

`SIZE` 可以是正整数或 `full`，默认是 100。

查看全部可用目标：

```bash
./run_models.sh --list-targets
```

### 单独模型

```bash
./run_models.sh qwen3_5_9b_ollama 100
./run_models.sh deepseek_v4_pro 100
```

### 五个小模型

```bash
./run_models.sh small 100
```

### 三个大模型

```bash
./run_models.sh large 100
```

### 大模型和小模型全集

```bash
./run_models.sh all 100
```

`all` 是论文主实验的 3 个大模型加 5 个小模型，不包含 thinking、flash 等消融配置。
模型组定义集中在：

```text
configs/models/model_groups.json
```

以后增加或调整论文模型池，只改这个配置，不改运行代码。

### 全量与其他数据类型

```bash
./run_models.sh small full
./run_models.sh large full
./run_models.sh all full

./run_models.sh all 100 --dataset-kind mixed
./run_models.sh large 100 --dataset-kind positive
```

### 自定义组合

```bash
./run_models.sh qwen3_5_9b_ollama,deepseek_v4_pro 100
```

### 检查而不运行

只打印最终命令：

```bash
./run_models.sh all 100 --show-command
```

执行 dry-run 管线，不调用模型：

```bash
./run_models.sh all 10 --dry-run
```

## 3. 运行与汇总严格分离

运行命令只做以下事情：

1. 选择或读取样本 manifest；
2. 调用模型；
3. 保存每个模型的原始结果、指标和运行配置。

它不会调用论文表格汇总器，也不会自动创建 `paper_tables/`。

运行完成以后，再独立汇总：

```bash
./summarize_latest_run.sh
```

要求本次 suite 的所有计划模型都完成：

```bash
./summarize_latest_run.sh --require-complete
```

或者指定运行目录：

```bash
python scripts/summarize_run.py \
  runs/zero_shot/<suite目录名> \
  --require-complete
```

汇总结果写入：

```text
runs/zero_shot/<suite>/paper_tables/
├── README.md
├── model_summary.csv
├── model_summary.json
├── l1_negative_over_edit.csv/.md
├── difficulty_negative_over_edit.csv/.md
└── register_negative_over_edit.csv/.md
```

## 4. 运行输出结构

同一次命令中的所有模型共享同一个 suite 和同一个 manifest：

```text
runs/zero_shot/<suite>/
├── sample_manifest.jsonl
├── suite_config.json
├── deepseek_v4_pro/
│   ├── results.jsonl
│   ├── results.csv
│   ├── metrics.json
│   ├── run_config.json
│   └── summary.txt
├── qwen3_7_plus/
│   └── ...
└── qwen3_5_9b_ollama/
    └── ...
```

需要后补模型时，复用已有 manifest：

```bash
./run_models.sh glm5_2_zhipu 100 \
  --manifest runs/zero_shot/<suite>/sample_manifest.jsonl \
  --run-dir runs/zero_shot/<suite>
```

这样新模型会写入同一个 suite，并与已有模型使用完全相同的样本。

## 5. 底层通用运行器

需要完整控制时，可以直接使用：

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 100 \
  --seed 42 \
  --models deepseek_v4_pro qwen3_7_plus
```

支持的主要参数：

- `--dataset-kind negative|positive|mixed`
- `--sample-size N` 或 `--full`
- `--manifest PATH`
- `--models MODEL_KEY ...`
- `--seed N`
- `--workers N`
- `--requests-per-second N`
- `--max-retries N`
- `--run-dir PATH`
- `--dry-run`

## 6. 数据抽样规则

- 默认只保留 `处置=主集`。
- 默认按 `L1类别` 分层抽样。
- 固定 `--seed` 后可复现。
- `mixed` 默认 Negative/Positive 为 1:1，可用 `--negative-ratio` 调整。
- 可用 `--split` 指定切分。
- 可用 `--exclude-controversial` 排除争议样本。
- 每次 suite 都保存实际 manifest，跨模型公平比较应复用该文件。

## 7. 指标口径

- `strict_changed`：只忽略首尾空白，其余字符变化均算修改。
- `normalized_changed`：额外忽略 Unicode 全半角和空白格式差异。
- Negative 主指标：`strict_over_edit_rate`、`content_over_edit_rate`。
- Positive：报告 edit trigger，不将其误称为改写成功率。
- Mixed 的 accuracy/F1 是编辑触发代理指标，不能替代去偏成功与语义保持评测。

## 8. 从 Excel 重新导出 JSONL

```bash
python scripts/export_main_jsonl.py \
  --negative /path/to/GoldenNegative.xlsx \
  --positive /path/to/GoldenPositive.xlsx \
  --output-dir data/source_jsonl
```

详细常用命令见 `command.md`。
