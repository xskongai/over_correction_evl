.PHONY: install export-baseline-data zero-shot test clean

install:
	pip install -e ".[dev]"

export-baseline-data:
	python scripts/export_main_jsonl.py \
	  --negative "$(NEG)" --positive "$(POS)" \
	  --output-dir data/source_jsonl

zero-shot:
	python scripts/run_zero_shot.py \
	  --dataset-kind $(or $(KIND),negative) \
	  --sample-size $(or $(N),100) \
	  --models $(or $(MODELS),deepseek_v4_flash) $(ARGS)

test:
	pytest

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .ruff_cache
