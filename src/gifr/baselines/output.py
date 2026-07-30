"""模型最终输出的最小、可审计后处理。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ProcessedOutput:
    raw: str
    scored: str
    applied: tuple[str, ...]


def process_output(raw: str, operations: Iterable[str]) -> ProcessedOutput:
    """只剥离明确的推理通道包装，不清理解释、引号或格式错误。

    论文指标应继续惩罚模型未遵守“只输出句子”的情况。因此这里只允许
    配置中显式声明、且不会改变最终答案语义的后处理。
    """
    text = raw
    applied: list[str] = []
    for operation in operations:
        if operation == "strip_closed_think_blocks":
            updated = _THINK_BLOCK.sub("", text).strip()
            if updated != text:
                applied.append(operation)
                text = updated
        else:
            raise ValueError(f"未知 output_postprocess 操作: {operation}")
    return ProcessedOutput(raw=raw, scored=text, applied=tuple(applied))
