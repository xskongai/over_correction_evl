# Model execution commands

模型运行与结果汇总完全分离：

- `run_models.sh`：只运行模型并保存原始结果。
- `summarize_latest_run.sh` / `scripts/summarize_run.py`：只汇总已有结果。

## 1. 查看可用模型组和 model key

```bash
./run_models.sh --list-targets
```

模型组在 `configs/models/model_groups.json` 中维护。

## 2. 单独运行一个模型

```bash
./run_models.sh qwen3_5_9b_ollama 100
```

任意配置中的 model key 都可以直接使用：

```bash
./run_models.sh deepseek_v4_pro 100
```

## 3. 运行全部小模型

```bash
./run_models.sh small 100
```

兼容旧命令：

```bash
./run_all_small_models.sh 100
```

## 4. 运行全部大模型

```bash
./run_models.sh large 100
```

## 5. 运行大模型和小模型全集

```bash
./run_models.sh all 100
```

`all` 指论文主实验的 3 个大模型和 5 个小模型，不包含 thinking/flash 等消融配置。
同一次命令中的模型共享同一个 `sample_manifest.jsonl`。

## 6. 跑全量数据

```bash
./run_models.sh small full
./run_models.sh large full
./run_models.sh all full
```

默认数据类型是 `negative`。其他数据类型通过参数覆盖：

```bash
./run_models.sh all 100 --dataset-kind mixed
./run_models.sh large 100 --dataset-kind positive
```

## 7. 指定多个自定义模型

使用逗号分隔：

```bash
./run_models.sh qwen3_5_9b_ollama,deepseek_v4_pro 100
```

## 8. Dry-run 和命令预览

只查看将运行什么，不创建运行目录：

```bash
./run_models.sh all 100 --show-command
```

执行完整管线但不调用真实模型：

```bash
./run_models.sh all 10 --dry-run
```

## 9. 单独汇总结果

汇总最新一次 suite：

```bash
./summarize_latest_run.sh
```

要求 suite 中计划运行的模型全部完成：

```bash
./summarize_latest_run.sh --require-complete
```

指定 suite：

```bash
python scripts/summarize_run.py \
  runs/zero_shot/<suite目录名> \
  --require-complete
```
