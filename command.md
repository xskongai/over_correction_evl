# v2.3 clean 数据重新测试命令

当前数据：

- Golden Negative：717 条（KEEP，用于 overcorrection / over-edit）
- Golden Positive：871 条（EDIT，用于 edit-trigger 与后续改写质量评估）
- 总计：1588 条

## 1. 进入项目

```bash
cd over_correction_eval_v2.3_clean
source .venv/bin/activate
```

## 2. 先做 dry-run，确认新数据已生效

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --sample-size 5 \
  --seed 42 \
  --models qwen3_5_9b_ollama \
  --dry-run
```

输出中应显示本次样本来自新的 `negative_main.jsonl`，dry-run 的 over-edit 应为 0。

## 3. 推荐：先重新跑五个小模型各 100 条

```bash
./run_all_small_models.sh 100
```

五个模型会共享同一份 `sample_manifest.jsonl`，便于直接比较：

- Qwen3.5-9B
- GLM4-9B
- DeepSeek-R1-8B
- Llama-3.1-8B
- Mistral-7B

## 4. 100 条确认无误后，跑 Golden Negative 全量 717 条

```bash
./run_all_small_models.sh full
```

## 5. 单独跑某个模型

```bash
python scripts/run_zero_shot.py \
  --dataset-kind negative \
  --full \
  --seed 42 \
  --workers 1 \
  --max-retries 0 \
  --models qwen3_5_9b_ollama
```

将最后一行模型 key 替换为：

```text
glm4_9b_ollama
deepseek_r1_8b_ollama
llama3_1_8b_ollama
mistral_7b_ollama
```

## 6. Positive 数据冒烟测试

```bash
python scripts/run_zero_shot.py \
  --dataset-kind positive \
  --sample-size 100 \
  --seed 42 \
  --workers 1 \
  --models qwen3_5_9b_ollama
```

注意：Positive 的 `edit_trigger_rate` 只表示模型是否改动了原句，不等同于改写正确率。
