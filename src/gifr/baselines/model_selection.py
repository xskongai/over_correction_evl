"""Resolve reusable model groups for experiment runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ModelSelectionError(ValueError):
    """Raised when a model target or group configuration is invalid."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelSelectionError(f"{label}不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelSelectionError(f"{label}不是有效 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelSelectionError(f"{label}顶层必须是 JSON object: {path}")
    return value


def load_model_keys(models_config: Path) -> list[str]:
    root = _load_json_object(models_config, "模型配置")
    models = root.get("models")
    if not isinstance(models, dict):
        raise ModelSelectionError(f"模型配置缺少 models object: {models_config}")
    return list(models)


def load_groups(groups_config: Path) -> dict[str, dict[str, Any]]:
    root = _load_json_object(groups_config, "模型分组配置")
    raw_groups = root.get("groups")
    if not isinstance(raw_groups, dict):
        raise ModelSelectionError(f"模型分组配置缺少 groups object: {groups_config}")

    groups: dict[str, dict[str, Any]] = {}
    for name, raw in raw_groups.items():
        if isinstance(raw, list):
            members = raw
            description = ""
        elif isinstance(raw, dict):
            members = raw.get("members")
            description = str(raw.get("description") or "")
        else:
            raise ModelSelectionError(f"group {name!r} 必须是 list 或 object")
        if not isinstance(members, list) or not all(isinstance(x, str) and x for x in members):
            raise ModelSelectionError(f"group {name!r} 的 members 必须是非空字符串列表")
        groups[str(name)] = {"members": list(members), "description": description}
    return groups


def _deduplicate(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def resolve_targets(
    targets: Iterable[str],
    *,
    model_keys: Iterable[str],
    groups: dict[str, dict[str, Any]],
) -> list[str]:
    """Resolve model keys and nested groups while preserving order.

    A group member beginning with ``@`` references another group. User targets may
    be model keys, group names, or comma-separated mixtures of both.
    """
    known_models = set(model_keys)
    expanded_targets: list[str] = []
    for raw_target in targets:
        expanded_targets.extend(part.strip() for part in raw_target.split(",") if part.strip())
    if not expanded_targets:
        raise ModelSelectionError("至少指定一个模型 key 或模型组")

    def resolve_one(target: str, stack: tuple[str, ...]) -> list[str]:
        group_name = target[1:] if target.startswith("@") else target
        if group_name in groups:
            if group_name in stack:
                cycle = " -> ".join((*stack, group_name))
                raise ModelSelectionError(f"模型组存在循环引用: {cycle}")
            resolved: list[str] = []
            for member in groups[group_name]["members"]:
                resolved.extend(resolve_one(member, (*stack, group_name)))
            return resolved
        if target.startswith("@"):
            raise ModelSelectionError(f"未知模型组: {group_name}")
        if target in known_models:
            return [target]
        available_groups = ", ".join(sorted(groups))
        raise ModelSelectionError(
            f"未知模型或模型组: {target}。可用模型组: {available_groups}"
        )

    resolved: list[str] = []
    for target in expanded_targets:
        resolved.extend(resolve_one(target, ()))
    return _deduplicate(resolved)
