"""Zero-shot 直接改写 baseline 指标。

Negative 上，输出发生变化即为 over-edit。
Positive 上，仅凭字符串变化只能测量 edit trigger / no-edit，不能证明改写正确。
Mixed 上给出 KEEP/EDIT 触发层面的代理决策指标。
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable


def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n <= 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"point": p, "low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def compute_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    data = list(rows)
    success = [r for r in data if r.get("status") == "completed"]
    failed = [r for r in data if r.get("status") == "failed"]
    negative = [r for r in success if r.get("gold_action") == "KEEP"]
    positive = [r for r in success if r.get("gold_action") == "EDIT"]

    overall = {
        "attempted": len(data),
        "completed": len(success),
        "failed": len(failed),
        "failure_rate": len(failed) / len(data) if data else 0.0,
        "negative_n": len(negative),
        "positive_n": len(positive),
        "mean_latency_seconds": _mean_optional(success, "latency_seconds"),
        "prompt_tokens": _sum_optional(success, "prompt_tokens"),
        "completion_tokens": _sum_optional(success, "completion_tokens"),
        "total_tokens": _sum_optional(success, "total_tokens"),
        "format_violation_rate": _rate(success, lambda r: bool(r.get("format_violation"))),
    }
    return {
        "overall": overall,
        "strict": _decision_metrics(success, changed_field="strict_changed"),
        "content": _decision_metrics(success, changed_field="normalized_changed"),
        "negative": _negative_metrics(negative),
        "positive": _positive_metrics(positive),
        "by_dataset_type": _group(success, "dataset_type"),
        "by_l1": _group(success, "L1类别"),
        "by_l2": _group(success, "L2子类"),
        "by_register": _group(success, "语体大类"),
        "by_difficulty": _group(success, "难度"),
    }


def _negative_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = sum(bool(r.get("strict_changed")) for r in rows)
    content = sum(bool(r.get("normalized_changed")) for r in rows)
    unchanged = len(rows) - strict
    format_only = strict - content
    return {
        "n": len(rows),
        "unchanged": unchanged,
        "format_only_changes": format_only,
        "content_changes": content,
        "keep_preservation": unchanged / len(rows) if rows else None,
        "strict_over_edit_rate": strict / len(rows) if rows else None,
        "strict_over_edit_ci95": wilson(strict, len(rows)) if rows else None,
        "content_over_edit_rate": content / len(rows) if rows else None,
        "content_over_edit_ci95": wilson(content, len(rows)) if rows else None,
        "format_only_rate": format_only / len(rows) if rows else None,
    }


def _positive_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = sum(bool(r.get("strict_changed")) for r in rows)
    content = sum(bool(r.get("normalized_changed")) for r in rows)
    return {
        "n": len(rows),
        "strict_edit_trigger_rate": strict / len(rows) if rows else None,
        "strict_no_edit_rate": (len(rows) - strict) / len(rows) if rows else None,
        "content_edit_trigger_rate": content / len(rows) if rows else None,
        "content_no_edit_rate": (len(rows) - content) / len(rows) if rows else None,
        "note": "字符串变化只能说明模型触发了编辑，不能证明偏见已被正确消除。",
    }


def _decision_metrics(rows: list[dict[str, Any]], *, changed_field: str) -> dict[str, Any]:
    tp = sum(r.get("gold_action") == "EDIT" and bool(r.get(changed_field)) for r in rows)
    fn = sum(r.get("gold_action") == "EDIT" and not bool(r.get(changed_field)) for r in rows)
    fp = sum(r.get("gold_action") == "KEEP" and bool(r.get(changed_field)) for r in rows)
    tn = sum(r.get("gold_action") == "KEEP" and not bool(r.get(changed_field)) for r in rows)
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "n": n,
        "confusion": {"tp_edit": tp, "fn_no_edit": fn, "fp_over_edit": fp, "tn_keep": tn},
        "action_accuracy_proxy": (tp + tn) / n if n else None,
        "edit_precision_proxy": precision,
        "edit_recall_proxy": recall,
        "edit_f1_proxy": f1,
        "keep_specificity": specificity,
        "balanced_accuracy_proxy": balanced,
        "note": "Positive 侧将任意内容变化视为 EDIT 触发，因此这些是动作代理指标，不是最终改写质量。",
    }


def _group(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if field == "dataset_type":
            value = str(row.get("dataset_type") or "未标注")
        else:
            value = str((row.get("metadata") or {}).get(field) or "未标注")
        buckets[value].append(row)
    out: list[dict[str, Any]] = []
    for name, group in sorted(buckets.items(), key=lambda item: item[0]):
        neg = [r for r in group if r.get("gold_action") == "KEEP"]
        pos = [r for r in group if r.get("gold_action") == "EDIT"]
        out.append({
            "group": name,
            "n": len(group),
            "negative_n": len(neg),
            "positive_n": len(pos),
            "negative_strict_over_edit_rate": _rate(neg, lambda r: bool(r.get("strict_changed"))),
            "negative_content_over_edit_rate": _rate(neg, lambda r: bool(r.get("normalized_changed"))),
            "positive_strict_edit_trigger_rate": _rate(pos, lambda r: bool(r.get("strict_changed"))),
            "positive_content_edit_trigger_rate": _rate(pos, lambda r: bool(r.get("normalized_changed"))),
        })
    return out


def _rate(rows: list[dict[str, Any]], pred) -> float | None:
    return sum(bool(pred(r)) for r in rows) / len(rows) if rows else None


def _sum_optional(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [int(r[key]) for r in rows if r.get(key) is not None]
    return sum(values) if values else None


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(values) if values else None
