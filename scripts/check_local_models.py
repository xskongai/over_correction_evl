#!/usr/bin/env python3
"""检查 Ollama 服务、模型是否已下载，并保存可复现的模型快照。"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "models" / "zero_shot_models.json"
DEFAULT_OUTPUT = ROOT / "runs" / "ollama_model_snapshot.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--models", nargs="*", default=None,
                   help="要检查的 model key；默认检查全部 runtime=ollama 配置")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    return p.parse_args()


def get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    args = parse_args()
    root = json.loads(args.models_config.read_text(encoding="utf-8"))
    all_models = root.get("models", {})
    keys = args.models or [k for k, v in all_models.items() if v.get("runtime") == "ollama"]
    unknown = [k for k in keys if k not in all_models]
    if unknown:
        print(f"未知 model key: {', '.join(unknown)}", file=sys.stderr)
        return 2

    try:
        tags = get_json(args.ollama_url.rstrip("/") + "/api/tags")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print("无法连接 Ollama。请先启动 Ollama，再重试。", file=sys.stderr)
        print(f"详情: {exc}", file=sys.stderr)
        return 2

    installed = {str(m.get("name")): m for m in tags.get("models", [])}
    rows: list[dict[str, Any]] = []
    missing = 0
    for key in keys:
        cfg = all_models[key]
        requested = str(cfg["model"])
        item = installed.get(requested)
        # Ollama 有时将无显式标签的名称规范化为 :latest。
        if item is None and ":" not in requested:
            item = installed.get(requested + ":latest")
        ok = item is not None
        missing += int(not ok)
        row = {
            "model_key": key,
            "requested_model": requested,
            "installed": ok,
            "runtime": cfg.get("runtime"),
            "model_family": cfg.get("model_family"),
            "parameter_count": cfg.get("parameter_count"),
            "configured_quantization": cfg.get("quantization"),
            "ollama": item or {},
        }
        rows.append(row)
        marker = "OK" if ok else "MISSING"
        detail = ""
        if item:
            details = item.get("details") or {}
            detail = f" digest={str(item.get('digest', ''))[:12]} quant={details.get('quantization_level', '?')}"
        print(f"[{marker:7}] {key:<34} -> {requested}{detail}")

    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ollama_url": args.ollama_url,
        "models_config": str(args.models_config.resolve()),
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n模型快照已保存: {args.output}")
    if missing:
        print(f"缺少 {missing} 个模型。先运行 pull 脚本或 ollama pull <model>。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
