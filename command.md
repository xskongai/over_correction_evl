# Model Execution Commands

Model execution and result summarization are completely separated:

- `run_models.sh`: only runs models and saves raw results.
- `summarize_latest_run.sh` / `scripts/summarize_run.py`: only summarizes existing results.

## 1. View Available Model Groups and Model Keys

```bash
./run_models.sh --list-targets
```

Model groups are maintained in `configs/models/model_groups.json`.

## 2. Run a Single Model

```bash
./run_models.sh qwen3_5_9b_ollama 100
```

Any model key defined in the configuration can be used directly:

```bash
./run_models.sh deepseek_v4_pro 100
```

## 3. Run All Small Models

```bash
./run_models.sh small 100
```

Compatible with the legacy command:

```bash
./run_all_small_models.sh 100
```

## 4. Run All Large Models

```bash
./run_models.sh large 100
```

## 5. Run All Large and Small Models

```bash
./run_models.sh all 100
```

`all` refers to the three large models and five small models used in the paper's main experiments. It does not include thinking/flash or other ablation configurations.

All models executed in the same command share the same `sample_manifest.jsonl`.

## 6. Run on the Full Dataset

```bash
./run_models.sh small full
./run_models.sh large full
./run_models.sh all full
```

The default dataset type is `negative`. Other dataset types can be specified with the following parameters:

```bash
./run_models.sh all 100 --dataset-kind mixed
./run_models.sh large 100 --dataset-kind positive
```

## 7. Specify Multiple Custom Models

Use comma-separated model keys:

```bash
./run_models.sh qwen3_5_9b_ollama,deepseek_v4_pro 100
```

## 8. Dry Run and Command Preview

Preview what will be executed without creating a run directory:

```bash
./run_models.sh all 100 --show-command
```

Run the complete pipeline without calling real models:

```bash
./run_models.sh all 10 --dry-run
```

## 9. Summarize Results Separately

Summarize the latest suite:

```bash
./summarize_latest_run.sh
```

Require all planned models in the suite to be completed:

```bash
./summarize_latest_run.sh --require-complete
```

Specify a suite explicitly:

```bash
python scripts/summarize_run.py \
  runs/zero_shot/<suite_directory_name> \
  --require-complete
```
