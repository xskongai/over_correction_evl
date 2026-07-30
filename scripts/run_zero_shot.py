#!/usr/bin/env python3
"""运行中文性别包容 zero-shot 直接改写 baseline。

核心判定：模型输出与原句不同，即认为模型触发了 EDIT。
- Golden Negative（gold=KEEP）：EDIT 即 over-edit。
- Golden Positive（gold=EDIT）：不变即 no-edit；但“发生变化”不等于改写正确。
- Mixed：报告 KEEP/EDIT 触发层面的代理分类指标。

同一次 suite 中的多个模型共享同一份 sample_manifest.jsonl，保证公平比较。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gifr.baselines.compare import compare_output
from gifr.baselines.dataset import (
    DatasetError,
    DatasetRecord,
    load_dataset,
    manifest_sha256,
    select_records,
    write_manifest,
)
from gifr.baselines.metrics import compute_metrics
from gifr.baselines.models import (
    ModelConfig,
    ModelConfigError,
    OpenAICompatibleClient,
    list_model_keys,
    load_dotenv,
    load_model_config,
)

DEFAULT_NEGATIVE = ROOT / "data" / "source_jsonl" / "negative_main.jsonl"
DEFAULT_POSITIVE = ROOT / "data" / "source_jsonl" / "positive_main.jsonl"
DEFAULT_MODELS = ROOT / "configs" / "models" / "zero_shot_models.json"
DEFAULT_PROMPT = ROOT / "configs" / "prompts" / "zero_shot_gender_inclusive_zh_v1.txt"


class RateLimiter:
    def __init__(self, requests_per_second: float | None) -> None:
        self.interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = self.next_time - now
            if delay > 0:
                time.sleep(delay)
            self.next_time = max(now, self.next_time) + self.interval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = p.add_argument_group("数据与抽样")
    source.add_argument("--negative", type=Path, default=DEFAULT_NEGATIVE)
    source.add_argument("--positive", type=Path, default=DEFAULT_POSITIVE)
    source.add_argument("--manifest", type=Path, default=None,
                        help="复用已冻结的 JSONL 样本清单；提供后忽略抽样参数")
    source.add_argument("--dataset-kind", choices=["negative", "positive", "mixed"],
                        default="negative")
    size = source.add_mutually_exclusive_group()
    size.add_argument("--sample-size", type=int, default=100,
                      help="默认 100；可设 300 等")
    size.add_argument("--full", action="store_true", help="运行筛选后的全量数据")
    source.add_argument("--seed", type=int, default=42)
    source.add_argument("--stratify-by", default="L1类别",
                        help="默认按 L1类别 分层；传 none 关闭")
    source.add_argument("--negative-ratio", type=float, default=0.5,
                        help="mixed 中 Negative 比例，默认 0.5")
    source.add_argument("--disposition", default="主集")
    source.add_argument("--split", default=None, help="只跑 index/dev/test 等指定切分")
    source.add_argument("--exclude-controversial", action="store_true")

    model = p.add_argument_group("模型")
    model.add_argument("--models", nargs="+", default=["deepseek_v4_flash"],
                       help="可一次指定多个 model key，共享同一抽样清单")
    model.add_argument("--models-config", type=Path, default=DEFAULT_MODELS)
    model.add_argument("--list-models", action="store_true")
    model.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    model.add_argument("--env-file", type=Path, default=ROOT / ".env")

    run = p.add_argument_group("运行")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--requests-per-second", type=float, default=None)
    run.add_argument("--max-retries", type=int, default=4)
    run.add_argument("--retry-base-seconds", type=float, default=1.5)
    run.add_argument("--checkpoint-every", type=int, default=10)
    run.add_argument("--dry-run", action="store_true",
                     help="不调用 API，直接回显原句，用于验证数据与指标管线")
    run.add_argument("--run-dir", type=Path, default=None)
    run.add_argument("--force-resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_models:
        for key in list_model_keys(args.models_config):
            print(key)
        return 0

    load_dotenv(args.env_file)
    try:
        models = [load_model_config(args.models_config, key) for key in args.models]
        prompt_template = args.prompt.read_text(encoding="utf-8")
        if "{sentence}" not in prompt_template:
            raise DatasetError(f"prompt 必须包含 {{sentence}} 占位符: {args.prompt}")
        records = _load_or_build_manifest(args)
    except (ModelConfigError, DatasetError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    suite_dir = (args.run_dir or _default_suite_dir(args, len(records))).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "sample_manifest.jsonl"
    suite_config_path = suite_dir / "suite_config.json"
    suite_config = _suite_config(args, records, models, prompt_template, manifest_path)
    if suite_config_path.exists():
        old_suite = json.loads(suite_config_path.read_text(encoding="utf-8"))
        if (old_suite.get("selection_fingerprint") != suite_config["selection_fingerprint"]
                and not args.force_resume):
            print(
                "错误: 该 suite run-dir 的样本或 prompt 与当前运行不一致。"
                "请换目录，或确认后加 --force-resume。",
                file=sys.stderr,
            )
            return 2
    write_manifest(manifest_path, records)
    suite_config_path.write_text(
        json.dumps(suite_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"样本类型: {args.dataset_kind if args.manifest is None else 'manifest'}")
    print(f"样本数量: {len(records)}  (KEEP={sum(r.gold_action == 'KEEP' for r in records)}, "
          f"EDIT={sum(r.gold_action == 'EDIT' for r in records)})")
    print(f"样本清单: {manifest_path}")
    print(f"模型: {', '.join(m.key for m in models)}")
    print(f"输出: {suite_dir}")

    exit_code = 0
    for model in models:
        try:
            code = _run_one_model(args, suite_dir, records, model, prompt_template)
            exit_code = max(exit_code, code)
        except KeyboardInterrupt:
            print("\n收到中断；已完成结果均已落盘。", file=sys.stderr)
            return 130
    _write_suite_summary(suite_dir, models)
    return exit_code


def _load_or_build_manifest(args: argparse.Namespace) -> list[DatasetRecord]:
    if args.manifest is not None:
        return load_dataset(args.manifest)
    negative = load_dataset(
        args.negative, default_dataset_type="negative", default_gold_action="KEEP"
    )
    positive = load_dataset(
        args.positive, default_dataset_type="positive", default_gold_action="EDIT"
    )
    stratify_by = None if str(args.stratify_by).lower() in {"none", "null", ""} else args.stratify_by
    return select_records(
        negative,
        positive,
        dataset_kind=args.dataset_kind,
        sample_size=None if args.full else args.sample_size,
        seed=args.seed,
        stratify_by=stratify_by,
        negative_ratio=args.negative_ratio,
        disposition=args.disposition,
        split=args.split,
        exclude_controversial=args.exclude_controversial,
    )


def _run_one_model(
    args: argparse.Namespace,
    suite_dir: Path,
    records: Sequence[DatasetRecord],
    model: ModelConfig,
    prompt_template: str,
) -> int:
    run_dir = suite_dir / model.key
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    config_path = run_dir / "run_config.json"
    run_config = _run_config(args, model, prompt_template, records)
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != run_config["fingerprint"] and not args.force_resume:
            raise RuntimeError(
                f"{model.key}: run-dir 中配置与当前运行不一致；请换目录或加 --force-resume"
            )
    else:
        config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    existing = _load_results(results_path)
    completed_ids = {r["id"] for r in existing if r.get("status") == "completed"}
    selected_ids = {r.id for r in records}
    todo = [r for r in records if r.id not in completed_ids]
    print("\n" + "=" * 78)
    print(f"模型 {model.key}: {model.model}  size={model.size_label}")
    print(f"已完成 {len(completed_ids & selected_ids)}/{len(records)}；待跑 {len(todo)}；"
          f"mode={'DRY' if args.dry_run else 'API'}")

    if not todo:
        _write_reports(run_dir, _load_results(results_path))
        print((run_dir / "summary.txt").read_text(encoding="utf-8"))
        return 0

    client = None if args.dry_run else OpenAICompatibleClient(
        model,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    limiter = RateLimiter(args.requests_per_second)
    append_lock = threading.Lock()
    total = len(todo)
    done_count = 0
    started = time.perf_counter()

    def run_one(record: DatasetRecord) -> dict[str, Any]:
        prompt = prompt_template.replace("{sentence}", record.text)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if args.dry_run:
            output = record.text
            latency = 0.0
            usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
            finish_reason = "dry_run"
            request_id = ""
            attempts = 0
            reasoning_chars = 0
            response_model = model.model
            system_fingerprint = "dry_run"
        else:
            assert client is not None
            limiter.wait()
            completion = client.complete(prompt)
            output = completion.content
            latency = completion.latency_seconds
            usage = {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
            }
            finish_reason = completion.finish_reason
            request_id = completion.request_id
            attempts = completion.attempts
            reasoning_chars = completion.reasoning_chars
            response_model = completion.response_model
            system_fingerprint = completion.system_fingerprint

        cmp = compare_output(record.text, output)
        return {
            "index": record.index,
            "id": record.id,
            "dataset_type": record.dataset_type,
            "gold_action": record.gold_action,
            "source": record.text,
            "output": output,
            "status": "completed",
            "comparison": cmp.label,
            "strict_changed": cmp.strict_changed,
            "normalized_changed": cmp.normalized_changed,
            "predicted_action_strict": "EDIT" if cmp.strict_changed else "KEEP",
            "predicted_action_content": "EDIT" if cmp.normalized_changed else "KEEP",
            "strict_action_correct_proxy": ("EDIT" if cmp.strict_changed else "KEEP") == record.gold_action,
            "content_action_correct_proxy": ("EDIT" if cmp.normalized_changed else "KEEP") == record.gold_action,
            "format_violation": cmp.format_violation,
            "format_violation_reasons": list(cmp.format_violation_reasons),
            "metadata": record.metadata,
            "model_key": model.key,
            "requested_model": model.model,
            "response_model": response_model,
            "system_fingerprint": system_fingerprint,
            "size_label": model.size_label,
            "parameter_count": model.parameter_count,
            "latency_seconds": round(latency, 4),
            **usage,
            "finish_reason": finish_reason,
            "request_id": request_id,
            "attempts": attempts,
            "reasoning_chars": reasoning_chars,
            "prompt_sha256": prompt_hash,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def failed_row(record: DatasetRecord, exc: BaseException) -> dict[str, Any]:
        return {
            "index": record.index,
            "id": record.id,
            "dataset_type": record.dataset_type,
            "gold_action": record.gold_action,
            "source": record.text,
            "output": "",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "metadata": record.metadata,
            "model_key": model.key,
            "requested_model": model.model,
            "size_label": model.size_label,
            "parameter_count": model.parameter_count,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures: dict[Future, DatasetRecord] = {
            executor.submit(run_one, record): record for record in todo
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                row = future.result()
            except BaseException as exc:  # 单条失败不终止批次
                row = failed_row(record, exc)
            with append_lock:
                _append_jsonl(results_path, row)
                done_count += 1
                elapsed = time.perf_counter() - started
                label = row.get("comparison") or "FAILED"
                gold = record.gold_action
                print(
                    f"[{done_count:>4}/{total}] {record.id:<12} gold={gold:<4} "
                    f"{label:<15} {elapsed:7.1f}s  {record.text[:34]}",
                    flush=True,
                )
                if args.checkpoint_every > 0 and done_count % args.checkpoint_every == 0:
                    _write_reports(run_dir, _load_results(results_path))

    _write_reports(run_dir, _load_results(results_path))
    print("\n" + (run_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0


def _default_suite_dir(args: argparse.Namespace, n: int) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    kind = "manifest" if args.manifest else args.dataset_kind
    return ROOT / "runs" / "zero_shot" / f"{stamp}_{kind}_{n}_seed{args.seed}"


def _suite_config(
    args: argparse.Namespace,
    records: Sequence[DatasetRecord],
    models: Sequence[ModelConfig],
    prompt_template: str,
    manifest_path: Path,
) -> dict[str, Any]:
    prompt_sha = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    manifest_sha = manifest_sha256(records)
    selection_fingerprint = hashlib.sha256(
        f"{manifest_sha}:{prompt_sha}:{args.dry_run}".encode("utf-8")
    ).hexdigest()
    return {
        "selection_fingerprint": selection_fingerprint,
        "dataset_kind": args.dataset_kind,
        "sample_size": len(records),
        "seed": args.seed,
        "stratify_by": args.stratify_by,
        "negative_ratio": args.negative_ratio,
        "disposition": args.disposition,
        "split": args.split,
        "exclude_controversial": args.exclude_controversial,
        "negative_n": sum(r.gold_action == "KEEP" for r in records),
        "positive_n": sum(r.gold_action == "EDIT" for r in records),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "models": [m.public_dict() for m in models],
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": prompt_sha,
        "dry_run": args.dry_run,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }


def _run_config(
    args: argparse.Namespace,
    model: ModelConfig,
    prompt_template: str,
    records: Sequence[DatasetRecord],
) -> dict[str, Any]:
    stable = {
        "manifest_sha256": manifest_sha256(records),
        "selected_count": len(records),
        "model": model.public_dict(),
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "dry_run": args.dry_run,
    }
    fingerprint = hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        **stable,
        "fingerprint": fingerprint,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"结果文件第 {line_no} 行损坏: {path}: {exc}") from exc
        by_id[str(row["id"])] = row
    return sorted(by_id.values(), key=lambda r: (int(r.get("index", 10**12)), str(r.get("id"))))


def _write_reports(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    metrics = compute_metrics(rows)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.txt").write_text(_render_summary(metrics), encoding="utf-8")
    _write_csv(run_dir / "results.csv", rows)


def _render_summary(metrics: dict[str, Any]) -> str:
    o = metrics["overall"]
    neg = metrics["negative"]
    pos = metrics["positive"]
    strict = metrics["strict"]
    lines = [
        "=" * 78,
        "Chinese Gender-Inclusive Zero-shot Rewrite Baseline",
        "=" * 78,
        f"Attempted                 {o['attempted']}",
        f"Completed                 {o['completed']}",
        f"Failed                    {o['failed']} ({o['failure_rate']:.2%})",
        f"Gold KEEP / EDIT           {o['negative_n']} / {o['positive_n']}",
        "-" * 78,
    ]
    if neg["n"]:
        ci = neg["strict_over_edit_ci95"]
        ci2 = neg["content_over_edit_ci95"]
        lines.extend([
            f"NEG KEEP preservation      {neg['keep_preservation']:.2%}",
            f"NEG strict over-edit       {neg['strict_over_edit_rate']:.2%} "
            f"(95% CI {ci['low']:.2%}–{ci['high']:.2%})",
            f"NEG content over-edit      {neg['content_over_edit_rate']:.2%} "
            f"(95% CI {ci2['low']:.2%}–{ci2['high']:.2%})",
            f"NEG format-only change     {neg['format_only_rate']:.2%}",
        ])
    if pos["n"]:
        lines.extend([
            f"POS strict edit trigger    {pos['strict_edit_trigger_rate']:.2%}",
            f"POS strict no-edit         {pos['strict_no_edit_rate']:.2%}",
            f"POS content edit trigger   {pos['content_edit_trigger_rate']:.2%}",
        ])
    if o["negative_n"] and o["positive_n"]:
        c = strict["confusion"]
        lines.extend([
            "-" * 78,
            f"Mixed action accuracy*     {_pct(strict['action_accuracy_proxy'])}",
            f"Mixed balanced accuracy*   {_pct(strict['balanced_accuracy_proxy'])}",
            f"Mixed edit precision/F1*   {_pct(strict['edit_precision_proxy'])} / "
            f"{_pct(strict['edit_f1_proxy'])}",
            f"Confusion TP/FN/FP/TN       {c['tp_edit']}/{c['fn_no_edit']}/"
            f"{c['fp_over_edit']}/{c['tn_keep']}",
        ])
    lines.extend([
        "-" * 78,
        f"Format violation rate      {_pct(o['format_violation_rate'])}",
        f"Mean latency seconds       {o['mean_latency_seconds']}",
        f"Total tokens               {o['total_tokens']}",
        "=" * 78,
        "strict：仅去除首尾空白；content：额外忽略 Unicode/空白格式差异。",
        "* Positive 只依据输出是否变化，因此 mixed 指标是编辑触发代理指标，",
        "  不能替代对去偏成功和语义保持的独立评测。",
    ])
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index", "id", "dataset_type", "gold_action", "status", "comparison",
        "predicted_action_strict", "predicted_action_content",
        "strict_action_correct_proxy", "content_action_correct_proxy",
        "strict_changed", "normalized_changed", "format_violation",
        "source", "output", "model_key", "requested_model", "response_model",
        "system_fingerprint", "size_label", "parameter_count", "latency_seconds",
        "prompt_tokens", "completion_tokens", "total_tokens", "finish_reason",
        "attempts", "reasoning_chars", "error_type", "error",
        "L1类别", "L2子类", "语体大类", "语体", "难度", "是否争议", "处置", "切分",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat.update(row.get("metadata") or {})
            writer.writerow(flat)


def _write_suite_summary(suite_dir: Path, models: Sequence[ModelConfig]) -> None:
    rows: list[dict[str, Any]] = []
    for model in models:
        path = suite_dir / model.key / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model_key": model.key,
            "model": model.model,
            "size_label": model.size_label,
            "parameter_count": model.parameter_count,
            "completed": m["overall"]["completed"],
            "failed": m["overall"]["failed"],
            "negative_n": m["overall"]["negative_n"],
            "positive_n": m["overall"]["positive_n"],
            "strict_over_edit_rate": m["negative"]["strict_over_edit_rate"],
            "content_over_edit_rate": m["negative"]["content_over_edit_rate"],
            "positive_edit_trigger_rate": m["positive"]["strict_edit_trigger_rate"],
            "mixed_action_accuracy_proxy": m["strict"]["action_accuracy_proxy"],
            "total_tokens": m["overall"]["total_tokens"],
        })
    (suite_dir / "model_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (suite_dir / "model_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
