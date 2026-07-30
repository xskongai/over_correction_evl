"""原句与模型输出的可复现比较规则。"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class CompareResult:
    source_stripped: str
    output_stripped: str
    strict_changed: bool
    normalized_changed: bool
    label: str
    format_violation: bool
    format_violation_reasons: tuple[str, ...]


def compare_output(source: str, output: str) -> CompareResult:
    """比较输出是否变化。

    primary/strict：只去掉首尾空白，任何其余字符变化都算过度改写。
    normalized：额外做 Unicode NFKC、换行和空白归一化，只作为诊断指标。
    不删除标点、不删除引号、不做语义相似度替代，避免把真实改写隐藏掉。
    """
    src = source.strip()
    out = output.strip()
    strict_changed = src != out
    normalized_changed = normalize_for_diagnostic(src) != normalize_for_diagnostic(out)
    if not strict_changed:
        label = "UNCHANGED"
    elif not normalized_changed:
        label = "FORMAT_ONLY"
    else:
        label = "CONTENT_CHANGED"
    # 原句本身可能以引号或类似前缀开头；只有模型改变输出时才判断是否新增包装。
    reasons = tuple(_format_violation_reasons(out)) if strict_changed else ()
    return CompareResult(
        source_stripped=src,
        output_stripped=out,
        strict_changed=strict_changed,
        normalized_changed=normalized_changed,
        label=label,
        format_violation=bool(reasons),
        format_violation_reasons=reasons,
    )


def normalize_for_diagnostic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def _format_violation_reasons(output: str) -> list[str]:
    reasons: list[str] = []
    if not output:
        reasons.append("empty_output")
        return reasons
    if output.startswith("```") or output.endswith("```"):
        reasons.append("code_fence")
    prefixes = (
        "原句：", "原句:", "改写：", "改写:", "修改后：", "修改后:",
        "输出：", "输出:", "无需修改：", "无需修改:", "无需修改，", "无需修改。",
    )
    if output.startswith(prefixes):
        reasons.append("explanatory_prefix")
    if len(output) >= 2 and (output[0], output[-1]) in {
        ("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"),
    }:
        reasons.append("outer_quotes")
    return reasons
