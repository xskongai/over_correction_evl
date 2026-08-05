# 全模型全量 Overcorrection 运行

## 1. 确认模型列表

```bash
./run_models.sh all full --show-command
```

确认输出中包含 `gemma2_9b_ollama`。

## 2. 运行全部模型 × 全量 Negative 数据

```bash
mkdir -p logs

caffeinate -dimsu ./run_models.sh all full \
  --run-dir runs/zero_shot/all_models_negative_full \
  2>&1 | tee -a "logs/all_models_negative_full_$(date +%Y%m%d_%H%M%S).log"
```

## 3. 中断后继续

重新执行上面同一条命令。固定的 `--run-dir` 会复用已有结果并补跑未完成样本。

## 4. 查看进度

```bash
tail -f "$(ls -t logs/all_models_negative_full_*.log | head -1)"
```

## 5. 完成后汇总

```bash
python scripts/summarize_run.py \
  runs/zero_shot/all_models_negative_full \
  --require-complete
```

检查失败记录：

```bash
grep -R '"status": "failed"' \
  runs/zero_shot/all_models_negative_full/*/results.jsonl
```

没有输出表示不存在失败记录。
