# Overcorrection Evaluation

This project evaluates **over-editing / overcorrection in Chinese gender-inclusive rewriting**.

## Core Decision Rule

After receiving the unified zero-shot instruction, if the model output differs from the original sentence, it is considered to have triggered `EDIT`.

- **Golden Negative:** the gold action is `KEEP`; substantive changes are counted as over-editing.
- **Golden Positive:** the gold action is `EDIT`; unchanged output is counted as no-edit. However, a change does **not** necessarily mean that the rewrite is correct.
- **Mixed:** reports proxy metrics at the KEEP/EDIT trigger level.

## Current Data

Only records with `Disposition = Main Set` in the cleaned workbooks are used:

- `data/source_jsonl/negative_main.jsonl`: 717 instances
- `data/source_jsonl/positive_main.jsonl`: 871 instances
- `data/source_jsonl/all_main.jsonl`: 1,588 instances

Formal runs read only the JSONL files; Excel workbooks are maintained as the manual source.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set the required API keys in `.env`:

```bash
DEEPSEEK_API_KEY=your_deepseek_key
DASHSCOPE_API_KEY=your_dashscope_key
ZHIPU_API_KEY=your_zhipu_key
```

Local models also require Ollama to be running. See `LOCAL_MODELS.md` for details. See `API_MODELS.md` for API-based models.

## 2. Unified Model Runner

The existing `scripts/run_zero_shot.py` already supports any number of models and ensures that they share the same `sample_manifest.jsonl`. A thin general-purpose entry point is provided on top of it:

```bash
./run_models.sh TARGET SIZE [other run_zero_shot.py arguments]
```

`TARGET` can be:

- a single model key;
- multiple comma-separated model keys;
- a model group such as `small`, `large`, or `all`.

`SIZE` can be a positive integer or `full`. The default is 100.

View all available targets:

```bash
./run_models.sh --list-targets
```

### Individual Models

```bash
./run_models.sh qwen3_5_9b_ollama 100
./run_models.sh deepseek_v4_pro 100
```

### Five Small Models

```bash
./run_models.sh small 100
```

### Three Large Models

```bash
./run_models.sh large 100
```

### All Large and Small Models

```bash
./run_models.sh all 100
```

`all` contains the three large models and five small models used in the main paper experiments. It does not include thinking, flash, or other ablation configurations.

Model group definitions are centralized in:

```text
configs/models/model_groups.json
```

To add or modify the paper's model pool, change only this configuration rather than the runner code.

### Full Runs and Other Dataset Types

```bash
./run_models.sh small full
./run_models.sh large full
./run_models.sh all full

./run_models.sh all 100 --dataset-kind mixed
./run_models.sh large 100 --dataset-kind positive
```

### Custom Model Combinations

```bash
./run_models.sh qwen3_5_9b_ollama,deepseek_v4_pro 100
```

### Inspect Without Running

Print only the final command:

```bash
./run_models.sh all 100 --show-command
```

Run the dry-run pipeline without calling any model:

```bash
./run_models.sh all 10 --dry-run
```

## 3. Strict Separation of Running and Summarization

The run command performs only the following operations:

1. Selects or reads the sample manifest;
2. Calls the models;
3. Saves the raw results, metrics, and run configuration for each model.

It does **not** call the paper table summarizer or automatically create `paper_tables/`.

After the run is complete, summarize the results separately:

```bash
./summarize_latest_run.sh
```

Require all planned models in the suite to be completed:

```bash
./summarize_latest_run.sh --require-complete
```

Alternatively, specify the run directory directly:

```bash
python scripts/summarize_run.py \
  runs/zero_shot/<suite_directory_name> \
  --require-complete
```

Summary outputs are written to:

```text
runs/zero_shot/<suite>/paper_tables/
├── README.md
├── model_summary.csv
├── model_summary.json
├── l1_negative_over_edit.csv/.md
├── difficulty_negative_over_edit.csv/.md
└── register_negative_over_edit.csv/.md
```

## 4. Run Output Structure

All models executed within the same command share the same suite and the same manifest:

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

When adding a model later, reuse the existing manifest:

```bash
./run_models.sh glm5_2_zhipu 100 \
  --manifest runs/zero_shot/<suite>/sample_manifest.jsonl \
  --run-dir runs/zero_shot/<suite>
```

This writes the new model into the same suite and ensures that it uses exactly the same samples as the existing models.

## 5. Low-Level General-Purpose Runner

For full control, use the underlying runner directly:

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 100 \
  --seed 42 \
  --models deepseek_v4_pro qwen3_7_plus
```

Main supported parameters:

- `--dataset-kind negative|positive|mixed`
- `--sample-size N` or `--full`
- `--manifest PATH`
- `--models MODEL_KEY ...`
- `--seed N`
- `--workers N`
- `--requests-per-second N`
- `--max-retries N`
- `--run-dir PATH`
- `--dry-run`

## 6. Data Sampling Rules

- By default, only records with `Disposition = Main Set` are retained.
- By default, stratified sampling is performed by `L1 Category`.
- Runs are reproducible when a fixed `--seed` is used.
- For `mixed`, the default Negative/Positive ratio is 1:1; use `--negative-ratio` to adjust it.
- Use `--split` to specify the data split.
- Each suite stores the actual manifest used. For fair cross-model comparison, reuse the same manifest.

## 7. Metric Definitions

- `strict_changed`: ignores only leading and trailing whitespace; all other character-level differences count as changes.
- `normalized_changed`: additionally ignores Unicode full-width/half-width differences and whitespace formatting differences.
- **Negative main metrics:** `strict_over_edit_rate`, `content_over_edit_rate`.
- **Positive:** reports the edit trigger rate and does not refer to it as a rewrite success rate.
- **Mixed:** accuracy/F1 are proxy metrics for edit-trigger classification and do not replace evaluation of debiasing success or semantic preservation.

## 8. Re-export JSONL from Excel

```bash
python scripts/export_main_jsonl.py \
  --negative /path/to/GoldenNegative.xlsx \
  --positive /path/to/GoldenPositive.xlsx \
  --output-dir data/source_jsonl
```