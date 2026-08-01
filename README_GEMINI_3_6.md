# Gemini 3.6 Flash migration

Copy this folder's contents into the project root, then run:

```bash
python scripts/migrate_gemini_3_6_flash.py
```

Verify the key is available:

```bash
./run_models.sh --list-targets
```

Run one sample first:

```bash
./run_models.sh gemini_3_6_flash 1
```

Run the same existing 100-sample manifest:

```bash
./run_models.sh gemini_3_6_flash 100 \
  --manifest runs/zero_shot/20260731T170832+0100_negative_100_seed42/sample_manifest.jsonl \
  --run-dir runs/zero_shot/20260731T170832+0100_negative_100_seed42
```
