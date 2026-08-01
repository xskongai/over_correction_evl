from __future__ import annotations

import json
from pathlib import Path

from gifr.baselines.compare import compare_output
from gifr.baselines.dataset import load_dataset, select_records
from gifr.baselines.metrics import compute_metrics
from gifr.baselines.models import load_model_config


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_compare_output_three_labels():
    assert compare_output("她很好。", "她很好。\n").label == "UNCHANGED"
    assert compare_output("编号 １２", "编号 12").label == "FORMAT_ONLY"
    assert compare_output("她很好。", "这个人很好。").label == "CONTENT_CHANGED"


def test_load_and_reproducible_mixed_sample(tmp_path: Path):
    neg_path = tmp_path / "negative.jsonl"
    pos_path = tmp_path / "positive.jsonl"
    neg_rows = [
        {"id": f"n{i}", "text": f"N{i}", "dataset_type": "negative", "gold_action": "KEEP",
         "L1类别": "A" if i < 5 else "B", "处置": "主集"}
        for i in range(10)
    ]
    pos_rows = [
        {"id": f"p{i}", "text": f"P{i}", "dataset_type": "positive", "gold_action": "EDIT",
         "L1类别": "C" if i < 5 else "D", "处置": "主集"}
        for i in range(10)
    ]
    _write(neg_path, neg_rows)
    _write(pos_path, pos_rows)
    neg = load_dataset(neg_path)
    pos = load_dataset(pos_path)
    a = select_records(neg, pos, dataset_kind="mixed", sample_size=10, seed=42)
    b = select_records(neg, pos, dataset_kind="mixed", sample_size=10, seed=42)
    assert [r.id for r in a] == [r.id for r in b]
    assert sum(r.gold_action == "KEEP" for r in a) == 5
    assert sum(r.gold_action == "EDIT" for r in a) == 5


def test_filter_main_and_controversial(tmp_path: Path):
    path = tmp_path / "negative.jsonl"
    _write(path, [
        {"id": "a", "text": "A", "dataset_type": "negative", "gold_action": "KEEP",
         "处置": "主集", "是否争议": "否"},
        {"id": "b", "text": "B", "dataset_type": "negative", "gold_action": "KEEP",
         "处置": "隔离·待确认", "是否争议": "否"},
        {"id": "c", "text": "C", "dataset_type": "negative", "gold_action": "KEEP",
         "处置": "主集", "是否争议": "是"},
    ])
    neg = load_dataset(path)
    selected = select_records(
        neg, [], dataset_kind="negative", sample_size=None, seed=1,
        disposition="主集", exclude_controversial=True,
    )
    assert [r.id for r in selected] == ["a"]


def test_metrics_negative_positive_mixed():
    rows = [
        {"status": "completed", "gold_action": "KEEP", "dataset_type": "negative",
         "strict_changed": False, "normalized_changed": False, "format_violation": False,
         "metadata": {}},
        {"status": "completed", "gold_action": "KEEP", "dataset_type": "negative",
         "strict_changed": True, "normalized_changed": True, "format_violation": False,
         "metadata": {}},
        {"status": "completed", "gold_action": "EDIT", "dataset_type": "positive",
         "strict_changed": True, "normalized_changed": True, "format_violation": False,
         "metadata": {}},
        {"status": "completed", "gold_action": "EDIT", "dataset_type": "positive",
         "strict_changed": False, "normalized_changed": False, "format_violation": False,
         "metadata": {}},
    ]
    metrics = compute_metrics(rows)
    assert metrics["negative"]["strict_over_edit_rate"] == 0.5
    assert metrics["positive"]["strict_edit_trigger_rate"] == 0.5
    assert metrics["strict"]["action_accuracy_proxy"] == 0.5


def test_model_config(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"m": {
        "provider": "openai_compatible",
        "base_url": "http://localhost:1/v1",
        "model": "demo"
    }}}), encoding="utf-8")
    cfg = load_model_config(path, "m")
    assert cfg.key == "m"
    assert cfg.model == "demo"


def test_model_config_base_url_env(tmp_path: Path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": {"m": {
        "provider": "openai_compatible",
        "base_url": "https://default.example/v1",
        "base_url_env": "TEST_MODEL_BASE_URL",
        "model": "demo"
    }}}), encoding="utf-8")
    monkeypatch.setenv("TEST_MODEL_BASE_URL", "https://override.example/v1/")
    cfg = load_model_config(path, "m")
    assert cfg.base_url == "https://override.example/v1"
    assert cfg.base_url_env == "TEST_MODEL_BASE_URL"

from gifr.baselines.model_selection import ModelSelectionError, resolve_targets


def test_resolve_single_group_and_all_models():
    models = ["large_a", "small_a", "small_b", "extra"]
    groups = {
        "small": {"members": ["small_a", "small_b"], "description": ""},
        "large": {"members": ["large_a"], "description": ""},
        "all": {"members": ["@large", "@small"], "description": ""},
    }
    assert resolve_targets(["small_a"], model_keys=models, groups=groups) == ["small_a"]
    assert resolve_targets(["small"], model_keys=models, groups=groups) == ["small_a", "small_b"]
    assert resolve_targets(["all"], model_keys=models, groups=groups) == [
        "large_a", "small_a", "small_b"
    ]


def test_resolve_comma_targets_deduplicates():
    models = ["a", "b"]
    groups = {"pair": {"members": ["a", "b"], "description": ""}}
    assert resolve_targets(["a,pair"], model_keys=models, groups=groups) == ["a", "b"]


def test_resolve_unknown_target_fails():
    try:
        resolve_targets(["missing"], model_keys=["a"], groups={})
    except ModelSelectionError as exc:
        assert "未知模型或模型组" in str(exc)
    else:
        raise AssertionError("expected ModelSelectionError")
