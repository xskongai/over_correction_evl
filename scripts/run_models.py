#!/usr/bin/env python3
"""Unified model runner for one model, a model group, or the full benchmark set.

This script only selects models and delegates execution to run_zero_shot.py.
It never invokes the paper-table summarizer.

Examples:
    python scripts/run_models.py qwen3_5_9b_ollama 100
    python scripts/run_models.py small 100
    python scripts/run_models.py large 100
    python scripts/run_models.py all full
    python scripts/run_models.py qwen3_5_9b_ollama,glm4_9b_ollama 20 --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gifr.baselines.model_selection import (  # noqa: E402
    ModelSelectionError,
    load_groups,
    load_model_keys,
    resolve_targets,
)

DEFAULT_MODELS_CONFIG = ROOT / "configs" / "models" / "zero_shot_models.json"
DEFAULT_GROUPS_CONFIG = ROOT / "configs" / "models" / "model_groups.json"
CORE_RUNNER = ROOT / "scripts" / "run_zero_shot.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "统一运行入口：TARGET 可为单个 model key、逗号分隔的多个 key，"
            "或 small / large / all 等模型组。其余参数原样传给 run_zero_shot.py。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  ./run_models.sh qwen3_5_9b_ollama 100\n"
            "  ./run_models.sh small 100\n"
            "  ./run_models.sh large 100\n"
            "  ./run_models.sh all 100\n"
            "  ./run_models.sh all full --dataset-kind mixed\n"
            "  ./run_models.sh small 10 --dry-run\n"
        ),
    )
    parser.add_argument("target", nargs="?", help="模型 key、逗号分隔 key，或模型组名")
    parser.add_argument("size", nargs="?", default="100", help="正整数或 full；默认 100")
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--groups-config", type=Path, default=DEFAULT_GROUPS_CONFIG)
    parser.add_argument("--list-targets", action="store_true", help="列出模型组和全部 model key")
    parser.add_argument("--show-command", action="store_true", help="只显示最终命令，不执行")
    return parser


def size_args(size: str) -> list[str]:
    if size == "full":
        return ["--full"]
    if size.isdigit() and int(size) > 0:
        return ["--sample-size", str(int(size))]
    raise ModelSelectionError("样本数必须是正整数或 full")


def print_targets(groups: dict[str, dict[str, object]], model_keys: list[str]) -> None:
    print("模型组:")
    for name, cfg in groups.items():
        description = str(cfg.get("description") or "")
        members = ", ".join(str(x) for x in cfg["members"])
        print(f"  {name:<10} {description}")
        print(f"             {members}")
    print("\n单个模型 key:")
    for key in model_keys:
        print(f"  {key}")


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    try:
        model_keys = load_model_keys(args.models_config)
        groups = load_groups(args.groups_config)
        if args.list_targets:
            print_targets(groups, model_keys)
            return 0
        if not args.target:
            parser.error("必须指定 TARGET；例如 small、large、all 或具体 model key")
        selected = resolve_targets([args.target], model_keys=model_keys, groups=groups)
        selected_size_args = size_args(args.size)
    except ModelSelectionError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if "--models" in passthrough:
        print("错误: run_models.py 已通过 TARGET 选择模型；不要再传 --models。", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        str(CORE_RUNNER),
        "--dataset-kind",
        "negative",
        *selected_size_args,
        "--seed",
        "42",
        "--workers",
        "1",
        "--models-config",
        str(args.models_config),
        "--models",
        *selected,
        *passthrough,
    ]

    print(f"目标: {args.target}")
    print(f"模型数: {len(selected)}")
    for index, key in enumerate(selected, start=1):
        print(f"  {index:>2}. {key}")
    print("\n执行命令:")
    print("  " + shlex.join(command))
    print()

    if args.show_command:
        return 0
    sys.stdout.flush()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
