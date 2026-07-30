#!/usr/bin/env python3
"""把人工维护的 Golden Negative/Positive Excel 导出为正式 JSONL。

正式 zero-shot 运行器不读取 Excel。本脚本只负责一次性数据准备，默认保留
``处置=主集`` 的句级数据，并统一写入 ``gold_action`` 与 ``dataset_type``。

示例：
    python scripts/export_main_jsonl.py \
      --negative /path/to/GoldenNegative.xlsx \
      --positive /path/to/GoldenPositive.xlsx \
      --output-dir data/source_jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import ZipFile

MAIN_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--negative", type=Path, required=True)
    p.add_argument("--positive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sheet", default="数据集")
    p.add_argument("--disposition", default="主集")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    neg_rows = _read_sheet(args.negative, args.sheet)
    pos_rows = _read_sheet(args.positive, args.sheet)
    negative = _convert(neg_rows, dataset_type="negative", gold_action="KEEP", disposition=args.disposition)
    positive = _convert(pos_rows, dataset_type="positive", gold_action="EDIT", disposition=args.disposition)

    neg_path = args.output_dir / "negative_main.jsonl"
    pos_path = args.output_dir / "positive_main.jsonl"
    all_path = args.output_dir / "all_main.jsonl"
    _write_jsonl(neg_path, negative)
    _write_jsonl(pos_path, positive)
    _write_jsonl(all_path, [*negative, *positive])

    report = {
        "negative_source": str(args.negative.resolve()),
        "positive_source": str(args.positive.resolve()),
        "sheet": args.sheet,
        "disposition": args.disposition,
        "negative_count": len(negative),
        "positive_count": len(positive),
        "total_count": len(negative) + len(positive),
        "outputs": {
            "negative": str(neg_path.resolve()),
            "positive": str(pos_path.resolve()),
            "all": str(all_path.resolve()),
        },
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _convert(
    rows: list[dict[str, Any]], *, dataset_type: str, gold_action: str, disposition: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("处置") or "").strip() != disposition:
            continue
        text = str(row.get("输入句子") or "").strip()
        if not text:
            continue
        rid = str(row.get("编号") or "").strip()
        item = {
            "id": rid,
            "text": text,
            "dataset_type": dataset_type,
            "gold_action": gold_action,
        }
        for key, value in row.items():
            if key in {"编号", "输入句子"}:
                continue
            item[key] = value
        out.append(item)
    ids = [x["id"] for x in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset_type} 导出后存在重复编号")
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_sheet(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with ZipFile(path) as zf:
        shared = _shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}
        target = None
        sheets = workbook.find("x:sheets", MAIN_NS)
        for sheet in list(sheets) if sheets is not None else []:
            if sheet.attrib.get("name") == sheet_name:
                target = _normalize_target(rels[sheet.attrib[REL_ID]])
                break
        if target is None:
            names = [s.attrib.get("name", "") for s in (list(sheets) if sheets is not None else [])]
            raise ValueError(f"找不到 sheet={sheet_name!r}；可选: {names}")
        root = ET.fromstring(zf.read(target))
        matrix: list[list[Any]] = []
        for row in root.findall(".//x:sheetData/x:row", MAIN_NS):
            cells: dict[int, Any] = {}
            max_col = -1
            for cell in row.findall("x:c", MAIN_NS):
                idx = _column_index(cell.attrib.get("r", "A1"))
                max_col = max(max_col, idx)
                cells[idx] = _cell_value(cell, shared)
            values = [None] * (max_col + 1)
            for idx, value in cells.items():
                values[idx] = value
            matrix.append(values)

    if not matrix:
        return []
    headers = [str(v or "").strip() for v in matrix[0]]
    rows: list[dict[str, Any]] = []
    for values in matrix[1:]:
        row = {
            header: values[i] if i < len(values) else None
            for i, header in enumerate(headers)
            if header
        }
        rows.append(row)
    return rows


def _shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall("x:si", MAIN_NS):
        out.append("".join((node.text or "") for node in si.iter(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
        )))
    return out


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("x:is", MAIN_NS)
        if inline is None:
            return None
        return "".join((node.text or "") for node in inline.iter(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
        ))
    value = cell.find("x:v", MAIN_NS)
    if value is None:
        return None
    raw = value.text or ""
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type == "b":
        return raw == "1"
    if cell_type == "str":
        return raw
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def _column_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


def _normalize_target(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


if __name__ == "__main__":
    raise SystemExit(main())
