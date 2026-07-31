# Dataset Update: v2.3 main-only clean

## New dataset size

| Dataset | Gold action | Previous project | v2.3 clean | Change |
|---|---|---:|---:|---:|
| Golden Negative | KEEP | 798 | 717 | -81 |
| Golden Positive | EDIT | 734 | 871 | +137 |
| Total | — | 1532 | 1588 | +56 |

## Source-level changes

- Negative: 716 IDs retained, 82 removed, 1 added (`NEG-1040`).
- Positive: 457 IDs retained, 277 removed, 414 added.
- For IDs retained in both old and new projects, the input text did not change.

## Data validation

All checks passed:

- no blank IDs or input sentences;
- no duplicate IDs;
- no duplicate input sentences within either dataset;
- no exact input-sentence overlap between Positive and Negative;
- every Negative row is labeled `无需改写` and its expected output equals the input;
- every Positive row is labeled `需改写`, has a reference rewrite, and the reference differs from the input;
- every row is `处置=主集`.

Machine-readable details are saved in:

```text
data/source_jsonl/data_validation_report.json
data/source_jsonl/dataset_summary.json
data/source_jsonl/export_report.json
```

## Recommended rerun order

```bash
./run_all_small_models.sh 100
./run_all_small_models.sh full
```

The five models share one manifest within each run, so their results are directly comparable.
