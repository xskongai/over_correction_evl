"""JSON/JSONL 数据读取、筛选与可复现抽样。

正式 zero-shot 实验只读取 JSON 或 JSONL。Excel 仅作为人工维护源，
可通过 ``scripts/export_main_jsonl.py`` 一次性导出。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


KEEP = "KEEP"
EDIT = "EDIT"


@dataclass(frozen=True)
class DatasetRecord:
    index: int
    id: str
    text: str
    gold_action: str
    dataset_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "text": self.text,
            "gold_action": self.gold_action,
            "dataset_type": self.dataset_type,
            **self.metadata,
        }


class DatasetError(ValueError):
    pass


def load_dataset(
    path: Path,
    *,
    default_dataset_type: str | None = None,
    default_gold_action: str | None = None,
) -> list[DatasetRecord]:
    """读取统一 JSON/JSONL 数据。

    推荐每条数据至少包含：``id``、``text``、``gold_action``、``dataset_type``。
    为兼容从工作簿直接导出的字段，也识别 ``编号``、``输入句子`` 和 ``期望标签``。
    """
    if path.suffix.lower() not in {".jsonl", ".json"}:
        raise DatasetError(f"正式实验仅支持 .jsonl/.json，不支持: {path.suffix}")
    if not path.exists():
        raise DatasetError(f"数据文件不存在: {path}")

    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"JSON 无效: {path}: {exc}") from exc
        if not isinstance(raw, list):
            raise DatasetError(".json 顶层必须是数组")
        items = raw
    else:
        items = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"JSONL 第 {line_no} 行无效: {exc}") from exc
            if not isinstance(item, dict):
                raise DatasetError(f"JSONL 第 {line_no} 行必须是对象")
            items.append(item)

    out: list[DatasetRecord] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise DatasetError(f"第 {idx + 1} 条数据必须是对象")
        text = str(item.get("text") or item.get("输入句子") or "").strip()
        if not text:
            continue
        rid = str(item.get("id") or item.get("编号") or f"row-{idx + 1}").strip()
        dataset_type = _normalize_dataset_type(
            item.get("dataset_type") or item.get("dataset") or default_dataset_type
        )
        gold_action = _normalize_gold_action(
            item.get("gold_action") or item.get("gold_label") or item.get("期望标签")
            or default_gold_action,
            dataset_type=dataset_type,
        )
        metadata: dict[str, Any] = {}
        nested = item.get("metadata")
        if isinstance(nested, dict):
            metadata.update({str(k): _jsonable(v) for k, v in nested.items()})
        reserved = {
            "id", "编号", "text", "输入句子", "gold_action", "gold_label", "期望标签",
            "dataset_type", "dataset", "metadata",
        }
        metadata.update({
            str(k): _jsonable(v) for k, v in item.items() if k not in reserved
        })
        metadata.setdefault("source_file", str(path))
        out.append(DatasetRecord(
            index=idx,
            id=rid,
            text=text,
            gold_action=gold_action,
            dataset_type=dataset_type,
            metadata=metadata,
        ))

    if not out:
        raise DatasetError(f"数据文件中没有有效文本: {path}")
    _check_unique_ids(out)
    return out


def select_records(
    negative: Sequence[DatasetRecord],
    positive: Sequence[DatasetRecord],
    *,
    dataset_kind: str,
    sample_size: int | None,
    seed: int,
    stratify_by: str | None = "L1类别",
    negative_ratio: float = 0.5,
    disposition: str | None = "主集",
    split: str | None = None,
    exclude_controversial: bool = False,
) -> list[DatasetRecord]:
    """筛选并抽取 negative / positive / mixed 样本。

    - negative 的 gold action 必须为 KEEP；
    - positive 的 gold action 必须为 EDIT；
    - mixed 在指定 sample_size 时默认按 1:1 抽取；
    - 同一 seed、同一数据文件和参数会得到完全相同的样本。
    """
    kind = dataset_kind.strip().lower()
    if kind not in {"negative", "positive", "mixed"}:
        raise DatasetError("dataset_kind 必须是 negative / positive / mixed")
    if sample_size is not None and sample_size <= 0:
        raise DatasetError("sample_size 必须大于 0，或不传表示全量")
    if not 0.0 <= negative_ratio <= 1.0:
        raise DatasetError("negative_ratio 必须在 [0, 1] 内")

    neg = _filter(
        [r for r in negative if r.gold_action == KEEP],
        disposition=disposition,
        split=split,
        exclude_controversial=exclude_controversial,
    )
    pos = _filter(
        [r for r in positive if r.gold_action == EDIT],
        disposition=disposition,
        split=split,
        exclude_controversial=exclude_controversial,
    )

    if kind == "negative":
        chosen = _stratified_sample(neg, sample_size, seed, stratify_by)
    elif kind == "positive":
        chosen = _stratified_sample(pos, sample_size, seed, stratify_by)
    elif sample_size is None:
        chosen = [*neg, *pos]
        random.Random(seed).shuffle(chosen)
    else:
        neg_n = int(round(sample_size * negative_ratio))
        neg_n = min(sample_size, max(0, neg_n))
        pos_n = sample_size - neg_n
        if neg_n > len(neg) or pos_n > len(pos):
            raise DatasetError(
                f"mixed 抽样数量不足：需要 negative={neg_n}, positive={pos_n}；"
                f"可用 negative={len(neg)}, positive={len(pos)}"
            )
        chosen = [
            *_stratified_sample(neg, neg_n, seed * 2 + 1, stratify_by),
            *_stratified_sample(pos, pos_n, seed * 2 + 2, stratify_by),
        ]
        random.Random(seed).shuffle(chosen)

    if not chosen:
        raise DatasetError("筛选后没有可运行样本")
    _check_unique_ids(chosen)
    return [
        DatasetRecord(
            index=i,
            id=r.id,
            text=r.text,
            gold_action=r.gold_action,
            dataset_type=r.dataset_type,
            metadata={**r.metadata, "selection_rank": i},
        )
        for i, r in enumerate(chosen)
    ]


def write_manifest(path: Path, records: Sequence[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def manifest_sha256(records: Sequence[DatasetRecord]) -> str:
    stable = "\n".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        for record in records
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _filter(
    records: Sequence[DatasetRecord],
    *,
    disposition: str | None,
    split: str | None,
    exclude_controversial: bool,
) -> list[DatasetRecord]:
    out: list[DatasetRecord] = []
    for record in records:
        meta = record.metadata
        if disposition is not None:
            got = str(meta.get("处置") or meta.get("disposition") or "").strip()
            # 已经导出的正式 JSONL 可以没有处置字段；有该字段时才执行筛选。
            if got and got != disposition:
                continue
        if split is not None:
            got = str(meta.get("切分") or meta.get("split") or "").strip()
            if got != split:
                continue
        if exclude_controversial:
            got = str(meta.get("是否争议") or meta.get("controversial") or "").strip().lower()
            if got in {"是", "yes", "true", "1"}:
                continue
        out.append(record)
    return out


def _stratified_sample(
    records: Sequence[DatasetRecord],
    n: int | None,
    seed: int,
    stratify_by: str | None,
) -> list[DatasetRecord]:
    records = list(records)
    if n is None or n >= len(records):
        out = list(records)
        random.Random(seed).shuffle(out)
        return out
    if n < 0:
        raise DatasetError("抽样数量不能为负")
    if not stratify_by:
        return random.Random(seed).sample(records, n)

    buckets: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        key = str(record.metadata.get(stratify_by) or "未标注")
        buckets[key].append(record)

    # 最大余数法分配各层名额，保证总数精确等于 n。
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, group in buckets.items():
        exact = n * len(group) / len(records)
        base = min(len(group), int(math.floor(exact)))
        quotas[key] = base
        remainders.append((exact - base, key))
    remaining = n - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if remaining <= 0:
            break
        if quotas[key] < len(buckets[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        for key in sorted(buckets):
            while remaining and quotas[key] < len(buckets[key]):
                quotas[key] += 1
                remaining -= 1

    out: list[DatasetRecord] = []
    for i, key in enumerate(sorted(buckets)):
        rnd = random.Random(seed + i * 1_000_003)
        out.extend(rnd.sample(buckets[key], quotas[key]))
    random.Random(seed).shuffle(out)
    return out


def _normalize_dataset_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "negative": "negative", "neg": "negative", "golden_negative": "negative",
        "positive": "positive", "pos": "positive", "golden_positive": "positive",
    }
    if text in mapping:
        return mapping[text]
    raise DatasetError(f"无法判断 dataset_type: {value!r}")


def _normalize_gold_action(value: Any, *, dataset_type: str) -> str:
    text = str(value or "").strip().upper()
    if text in {KEEP, "CLEAR", "无需改写", "无需修改", "NEGATIVE"}:
        return KEEP
    if text in {EDIT, "FLAG", "需改写", "需要改写", "POSITIVE"}:
        return EDIT
    if not text:
        return KEEP if dataset_type == "negative" else EDIT
    raise DatasetError(f"无法判断 gold_action: {value!r}")


def _check_unique_ids(records: Iterable[DatasetRecord]) -> None:
    seen: set[str] = set()
    dup: list[str] = []
    for record in records:
        if record.id in seen:
            dup.append(record.id)
        seen.add(record.id)
    if dup:
        raise DatasetError(f"数据集中存在重复编号: {', '.join(dup[:10])}")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
