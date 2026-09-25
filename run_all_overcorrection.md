# Full-Scale Overcorrection Run for All Models

## 1. Confirm the Model List

```bash
./run_models.sh all full --show-command
```

Confirm that the output includes `gemma2_9b_ollama`.

## 2. Run All Models × Full Negative Dataset

```bash
mkdir -p logs

caffeinate -dimsu ./run_models.sh all full \
  --run-dir runs/zero_shot/all_models_negative_full \
  2>&1 | tee -a "logs/all_models_negative_full_$(date +%Y%m%d_%H%M%S).log"
```

## 3. Resume After Interruption

Run the same command above again. The fixed `--run-dir` will reuse existing results and rerun only incomplete samples.

## 4. Check Progress

```bash
tail -f "$(ls -t logs/all_models_negative_full_*.log | head -1)"
```

## 5. Summarize After Completion

```bash
python scripts/summarize_run.py \
  runs/zero_shot/all_models_negative_full \
  --require-complete
```

Check for failed records:

```bash
grep -R '"status": "failed"' \
  runs/zero_shot/all_models_negative_full/*/results.jsonl
```

No output means that there are no failed records.
