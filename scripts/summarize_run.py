#!/usr/bin/env python3
"""Summarize one zero-shot evaluation suite into paper-friendly tables.

Typical usage:
    python scripts/summarize_run.py
    python scripts/summarize_run.py runs/zero_shot/20260731T151151+0100_negative_100_seed42

When RUN_DIR is omitted, the newest suite under runs/zero_shot is selected.
The script reads each model subdirectory's metrics.json and run_config.json and
writes publication-ready Markdown/CSV tables under RUN_DIR/paper_tables/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "zero_shot"

PRETTY_MODEL_NAMES = {
    "qwen3_5_9b_ollama": "Qwen3.5-9B",
    "glm4_9b_ollama": "GLM-4-9B",
    "deepseek_r1_8b_ollama": "DeepSeek-R1-8B",
    "llama3_1_8b_ollama": "Llama-3.1-8B",
    "mistral_7b_ollama": "Mistral-7B",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "deepseek_v4_pro": "DeepSeek-V4-Pro",
    "qwen3_7_flash": "Qwen3.7-Flash",
    "qwen3_7_plus": "Qwen3.7-Plus",
    "qwen3_7_max": "Qwen3.7-Max",
    "glm5_2_zhipu": "GLM-5.2",
    "glm5_2_dashscope": "GLM-5.2",
    "gpt_4o": "GPT-4o",
    "gemini_3_6_flash": "Gemini-3.6-Flash",
    "gemini_2_5_pro": "Gemini-2.5-Pro",
    "gemini_2_5_flash": "Gemini-2.5-Flash",
}


@dataclass(frozen=True)
class ModelResult:
    model_key: str
    display_name: str
    requested_model: str
    parameter_count: str
    metrics: dict[str, Any]
    run_config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a zero-shot suite into paper-friendly Markdown and CSV tables."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Suite directory. Omit to summarize the newest directory under runs/zero_shot.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Root used when RUN_DIR is omitted (default: runs/zero_shot).",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: RUN_DIR/paper_tables).",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if one or more models listed in suite_config.json have no metrics.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def resolve_run_dir(run_dir_arg: str | None, runs_root_arg: str) -> Path:
    if run_dir_arg:
        candidate = Path(run_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not (candidate / "suite_config.json").is_file():
            raise RuntimeError(f"Not a suite directory (suite_config.json missing): {candidate}")
        return candidate

    root = Path(runs_root_arg).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    candidates = [p.parent for p in root.glob("*/suite_config.json") if p.is_file()]
    if not candidates:
        raise RuntimeError(f"No completed or in-progress suite found under: {root}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def expected_model_keys(suite_config: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for model in suite_config.get("models", []):
        if isinstance(model, dict) and model.get("key"):
            keys.append(str(model["key"]))
    return keys


def prettify_model_name(model_key: str, requested_model: str) -> str:
    if model_key in PRETTY_MODEL_NAMES:
        return PRETTY_MODEL_NAMES[model_key]
    if requested_model:
        name = requested_model.replace(":latest", "").replace(":", "-")
        return name
    return model_key.replace("_ollama", "").replace("_", "-")


def collect_model_results(run_dir: Path, expected: list[str]) -> tuple[list[ModelResult], list[str]]:
    discovered: dict[str, ModelResult] = {}
    for metrics_path in sorted(run_dir.glob("*/metrics.json")):
        model_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        run_config_path = model_dir / "run_config.json"
        run_config = load_json(run_config_path) if run_config_path.is_file() else {}
        model_cfg = run_config.get("model") if isinstance(run_config.get("model"), dict) else {}
        model_key = str(model_cfg.get("key") or model_dir.name)
        requested_model = str(model_cfg.get("model") or model_key)
        parameter_count = str(model_cfg.get("parameter_count") or "—")
        discovered[model_key] = ModelResult(
            model_key=model_key,
            display_name=prettify_model_name(model_key, requested_model),
            requested_model=requested_model,
            parameter_count=parameter_count,
            metrics=metrics,
            run_config=run_config,
        )

    order = expected + sorted(k for k in discovered if k not in expected)
    results = [discovered[k] for k in order if k in discovered]
    missing = [k for k in expected if k not in discovered]
    return results, missing


def nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pct_number(value: Any, digits: int = 2) -> float | None:
    number = finite_number(value)
    return round(number * 100, digits) if number is not None else None


def pct_text(value: Any, digits: int = 2) -> str:
    number = pct_number(value, digits)
    return "—" if number is None else f"{number:.{digits}f}%"


def ci_text(ci: Any, digits: int = 2) -> str:
    if not isinstance(ci, dict):
        return "—"
    low = pct_number(ci.get("low"), digits)
    high = pct_number(ci.get("high"), digits)
    if low is None or high is None:
        return "—"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def num_text(value: Any, digits: int = 2) -> str:
    number = finite_number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def integer_text(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(escape_md(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(v) for v in row) + " |")
    return "\n".join(lines)


def build_compact_table(results: list[ModelResult]) -> tuple[list[str], list[list[str]]]:
    has_negative = any((nested(r.metrics, "overall", "negative_n", default=0) or 0) > 0 for r in results)
    has_positive = any((nested(r.metrics, "overall", "positive_n", default=0) or 0) > 0 for r in results)

    headers = ["Model", "Params", "Completed"]
    if has_negative:
        headers += [
            "NEG KEEP ↑",
            "NEG Content Over-edit ↓",
            "95% CI",
            "NEG Strict Over-edit ↓",
        ]
    if has_positive:
        headers += ["POS Content Edit Trigger ↑"]
    if has_negative and has_positive:
        headers += ["Content Action Acc. ↑"]
    headers += ["Format Violation ↓", "Failed"]

    rows: list[list[str]] = []
    for result in results:
        metrics = result.metrics
        overall = metrics.get("overall", {})
        negative = metrics.get("negative", {})
        positive = metrics.get("positive", {})
        content = metrics.get("content", {})
        row = [
            result.display_name,
            result.parameter_count,
            integer_text(overall.get("completed")),
        ]
        if has_negative:
            row += [
                pct_text(negative.get("keep_preservation")),
                pct_text(negative.get("content_over_edit_rate")),
                ci_text(negative.get("content_over_edit_ci95")),
                pct_text(negative.get("strict_over_edit_rate")),
            ]
        if has_positive:
            row += [pct_text(positive.get("content_edit_trigger_rate"))]
        if has_negative and has_positive:
            row += [pct_text(content.get("action_accuracy_proxy"))]
        row += [
            pct_text(overall.get("format_violation_rate")),
            integer_text(overall.get("failed")),
        ]
        rows.append(row)
    return headers, rows


def build_detailed_rows(results: list[ModelResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result.metrics
        overall = metrics.get("overall", {})
        negative = metrics.get("negative", {})
        positive = metrics.get("positive", {})
        strict = metrics.get("strict", {})
        content = metrics.get("content", {})
        strict_ci = negative.get("strict_over_edit_ci95") or {}
        content_ci = negative.get("content_over_edit_ci95") or {}
        rows.append({
            "model": result.display_name,
            "model_key": result.model_key,
            "requested_model": result.requested_model,
            "params": result.parameter_count,
            "attempted": overall.get("attempted"),
            "completed": overall.get("completed"),
            "failed": overall.get("failed"),
            "failure_rate_pct": pct_number(overall.get("failure_rate")),
            "negative_n": overall.get("negative_n"),
            "positive_n": overall.get("positive_n"),
            "neg_keep_preservation_pct": pct_number(negative.get("keep_preservation")),
            "neg_strict_over_edit_pct": pct_number(negative.get("strict_over_edit_rate")),
            "neg_strict_ci95_low_pct": pct_number(strict_ci.get("low")),
            "neg_strict_ci95_high_pct": pct_number(strict_ci.get("high")),
            "neg_content_over_edit_pct": pct_number(negative.get("content_over_edit_rate")),
            "neg_content_ci95_low_pct": pct_number(content_ci.get("low")),
            "neg_content_ci95_high_pct": pct_number(content_ci.get("high")),
            "neg_format_only_change_pct": pct_number(negative.get("format_only_rate")),
            "pos_strict_edit_trigger_pct": pct_number(positive.get("strict_edit_trigger_rate")),
            "pos_content_edit_trigger_pct": pct_number(positive.get("content_edit_trigger_rate")),
            "strict_action_accuracy_proxy_pct": pct_number(strict.get("action_accuracy_proxy")),
            "content_action_accuracy_proxy_pct": pct_number(content.get("action_accuracy_proxy")),
            "content_balanced_accuracy_proxy_pct": pct_number(content.get("balanced_accuracy_proxy")),
            "format_violation_pct": pct_number(overall.get("format_violation_rate")),
            "mean_latency_seconds": round(float(overall["mean_latency_seconds"]), 4)
            if finite_number(overall.get("mean_latency_seconds")) is not None else None,
            "prompt_tokens": overall.get("prompt_tokens"),
            "completion_tokens": overall.get("completion_tokens"),
            "total_tokens": overall.get("total_tokens"),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_map(result: ModelResult, key: str) -> dict[str, dict[str, Any]]:
    entries = result.metrics.get(key, [])
    if not isinstance(entries, list):
        return {}
    return {
        str(entry.get("group", "未标注")): entry
        for entry in entries
        if isinstance(entry, dict)
    }


def group_sort_key(name: str) -> tuple[int, str]:
    for prefix in ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十"):
        if name.startswith(prefix):
            return (("一二三四五六七八九十".index(prefix)), name)
    return (999, name)


def build_group_rows(
    results: list[ModelResult],
    metric_group_key: str,
    value_key: str,
) -> list[dict[str, Any]]:
    per_model = {r.model_key: group_map(r, metric_group_key) for r in results}
    group_names = sorted(
        {name for groups in per_model.values() for name in groups},
        key=group_sort_key,
    )
    output: list[dict[str, Any]] = []
    for group_name in group_names:
        n_values: list[int] = []
        row: dict[str, Any] = {"group": group_name}
        model_values: list[float] = []
        for result in results:
            entry = per_model[result.model_key].get(group_name, {})
            n_value = entry.get("negative_n") if "negative" in value_key else entry.get("positive_n")
            if n_value is not None:
                try:
                    n_values.append(int(n_value))
                except (TypeError, ValueError):
                    pass
            value = pct_number(entry.get(value_key))
            row[result.display_name] = value
            if value is not None:
                model_values.append(value)
        row["n"] = max(n_values) if n_values else None
        row["mean_pct"] = round(sum(model_values) / len(model_values), 2) if model_values else None
        ordered = {"group": row.pop("group"), "n": row.pop("n")}
        ordered.update(row)
        output.append(ordered)
    return output


def group_rows_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    formatted: list[list[str]] = []
    for row in rows:
        current: list[str] = []
        for header in headers:
            value = row.get(header)
            if header == "group":
                current.append(str(value))
            elif header == "n":
                current.append(integer_text(value))
            else:
                current.append("—" if value is None else f"{float(value):.2f}%")
        formatted.append(current)
    display_headers = ["Category" if h == "group" else "N" if h == "n" else "Mean" if h == "mean_pct" else h for h in headers]
    return markdown_table(display_headers, formatted)


def write_group_outputs(
    output_dir: Path,
    results: list[ModelResult],
    suite_config: dict[str, Any],
) -> list[str]:
    generated: list[str] = []
    has_negative = any((nested(r.metrics, "overall", "negative_n", default=0) or 0) > 0 for r in results)
    has_positive = any((nested(r.metrics, "overall", "positive_n", default=0) or 0) > 0 for r in results)
    breakdowns = [
        ("by_l1", "l1"),
        ("by_difficulty", "difficulty"),
        ("by_register", "register"),
    ]
    for metric_key, file_stem in breakdowns:
        if has_negative:
            rows = build_group_rows(results, metric_key, "negative_content_over_edit_rate")
            if rows:
                csv_path = output_dir / f"{file_stem}_negative_over_edit.csv"
                md_path = output_dir / f"{file_stem}_negative_over_edit.md"
                write_csv(csv_path, rows)
                md_path.write_text(
                    f"# {file_stem.upper()} breakdown: negative content over-edit rate\n\n"
                    + group_rows_markdown(rows)
                    + "\n",
                    encoding="utf-8",
                )
                generated.extend([csv_path.name, md_path.name])
        if has_positive:
            rows = build_group_rows(results, metric_key, "positive_content_edit_trigger_rate")
            if rows:
                csv_path = output_dir / f"{file_stem}_positive_edit_trigger.csv"
                md_path = output_dir / f"{file_stem}_positive_edit_trigger.md"
                write_csv(csv_path, rows)
                md_path.write_text(
                    f"# {file_stem.upper()} breakdown: positive content edit-trigger rate\n\n"
                    + group_rows_markdown(rows)
                    + "\n",
                    encoding="utf-8",
                )
                generated.extend([csv_path.name, md_path.name])
    return generated


def integrity_warnings(
    suite_config: dict[str, Any],
    results: list[ModelResult],
) -> list[str]:
    warnings: list[str] = []
    expected_total = suite_config.get("sample_size")
    try:
        expected_total_int = int(expected_total)
    except (TypeError, ValueError):
        expected_total_int = None
    suite_manifest = str(suite_config.get("manifest_sha256") or "")

    for result in results:
        overall = result.metrics.get("overall", {})
        attempted = overall.get("attempted")
        completed = overall.get("completed")
        failed = overall.get("failed")
        try:
            attempted_int = int(attempted)
        except (TypeError, ValueError):
            attempted_int = None
        try:
            completed_int = int(completed)
            failed_int = int(failed)
        except (TypeError, ValueError):
            completed_int = failed_int = None

        if expected_total_int is not None and attempted_int != expected_total_int:
            warnings.append(
                f"{result.model_key}: attempted={attempted!r}, expected={expected_total_int}"
            )
        if (
            attempted_int is not None
            and completed_int is not None
            and failed_int is not None
            and completed_int + failed_int != attempted_int
        ):
            warnings.append(
                f"{result.model_key}: completed+failed does not equal attempted "
                f"({completed_int}+{failed_int}!={attempted_int})"
            )
        model_manifest = str(result.run_config.get("manifest_sha256") or "")
        if suite_manifest and model_manifest and model_manifest != suite_manifest:
            warnings.append(f"{result.model_key}: manifest SHA-256 differs from suite")
    return warnings


def build_readme(
    run_dir: Path,
    suite_config: dict[str, Any],
    results: list[ModelResult],
    missing: list[str],
    integrity: list[str],
    headers: list[str],
    compact_rows: list[list[str]],
) -> str:
    expected_n = len(expected_model_keys(suite_config))
    dataset_kind = suite_config.get("dataset_kind", "unknown")
    sample_size = suite_config.get("sample_size", "unknown")
    seed = suite_config.get("seed", "unknown")
    negative_n = suite_config.get("negative_n", "unknown")
    positive_n = suite_config.get("positive_n", "unknown")
    manifest_sha = str(suite_config.get("manifest_sha256") or "unknown")
    prompt_sha = str(suite_config.get("prompt_sha256") or "unknown")

    lines = [
        "# Paper-ready evaluation summary",
        "",
        f"- **Run:** `{run_dir.name}`",
        f"- **Dataset kind:** `{dataset_kind}`",
        f"- **Sample size:** `{sample_size}` (NEG={negative_n}, POS={positive_n})",
        f"- **Seed:** `{seed}`",
        f"- **Models summarized:** `{len(results)}/{expected_n or len(results)}`",
        f"- **Manifest SHA-256:** `{manifest_sha}`",
        f"- **Prompt SHA-256:** `{prompt_sha}`",
        "",
        "## Main results",
        "",
        markdown_table(headers, compact_rows),
        "",
        "**Metric note.** NEG content over-edit ignores Unicode/whitespace-only differences; "
        "NEG strict over-edit counts any output-string change. POS edit trigger is only a "
        "trigger proxy and does not establish rewrite correctness or semantic preservation.",
    ]
    if missing:
        lines += [
            "",
            "**Incomplete suite warning.** Missing metrics for: " + ", ".join(f"`{m}`" for m in missing) + ".",
        ]
    if integrity:
        lines += [
            "",
            "**Integrity warning.** Do not use this table as a same-sample model comparison until these issues are resolved:",
        ]
        lines += [f"- {item}" for item in integrity]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        run_dir = resolve_run_dir(args.run_dir, args.runs_root)
        suite_config = load_json(run_dir / "suite_config.json")
        expected = expected_model_keys(suite_config)
        results, missing = collect_model_results(run_dir, expected)
        if not results:
            raise RuntimeError(f"No model metrics.json found in: {run_dir}")
        integrity = integrity_warnings(suite_config, results)
        if args.require_complete and (missing or integrity):
            problems = []
            if missing:
                problems.append("missing metrics for: " + ", ".join(missing))
            if integrity:
                problems.append("integrity issues: " + "; ".join(integrity))
            raise RuntimeError("Suite is incomplete or inconsistent; " + " | ".join(problems))

        output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir / "paper_tables"
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        headers, compact_rows = build_compact_table(results)
        detailed_rows = build_detailed_rows(results)

        readme_path = output_dir / "README.md"
        readme_path.write_text(
            build_readme(run_dir, suite_config, results, missing, integrity, headers, compact_rows),
            encoding="utf-8",
        )
        write_csv(output_dir / "model_summary.csv", detailed_rows)

        raw_summary = {
            "run_dir": str(run_dir),
            "suite_config": suite_config,
            "missing_models": missing,
            "integrity_warnings": integrity,
            "models": [
                {
                    "model_key": r.model_key,
                    "display_name": r.display_name,
                    "requested_model": r.requested_model,
                    "parameter_count": r.parameter_count,
                    "metrics": r.metrics,
                }
                for r in results
            ],
        }
        (output_dir / "model_summary.json").write_text(
            json.dumps(raw_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        generated = ["README.md", "model_summary.csv", "model_summary.json"]
        generated += write_group_outputs(output_dir, results, suite_config)

        print("=" * 96)
        print(f"Run: {run_dir}")
        print(f"Models summarized: {len(results)}/{len(expected) or len(results)}")
        if missing:
            print("Missing models: " + ", ".join(missing))
        if integrity:
            print("Integrity warnings:")
            for item in integrity:
                print(f"  - {item}")
        print("-" * 96)
        print(markdown_table(headers, compact_rows))
        print("-" * 96)
        print(f"Paper tables written to: {output_dir}")
        print("Files: " + ", ".join(generated))
        print("=" * 96)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
