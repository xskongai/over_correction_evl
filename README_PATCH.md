# GPT-4o + Gemini integration patch

将本目录内容复制或解压到现有 `over_correction_eval` 项目根目录，然后运行：

```bash
python scripts/add_gpt4o_gemini_models.py
```

脚本会：

- 新增 `gpt_4o`、`gemini_2_5_pro`、`gemini_2_5_flash`；
- 把 GPT-4o 和 Gemini 2.5 Pro 追加到现有 `large` 组；
- 新增 `openai_google` 组；
- 保留你现有 large/small/all 成员，不覆盖已有模型；
- 为汇总表增加易读模型名。

在 `.env` 中填写：

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

冒烟测试：

```bash
./run_models.sh gpt_4o 1
./run_models.sh gemini_2_5_pro 1
```

一起跑 100 条：

```bash
./run_models.sh openai_google 100
```

加入全部模型：

```bash
./run_models.sh all 100
```
