#!/usr/bin/env python3
"""Migrate the project from Gemini 2.5 configs to Gemini 3.6 Flash.

Run from the project root:
    python scripts/migrate_gemini_3_6_flash.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "configs/models/zero_shot_models.json"
GROUPS = ROOT / "configs/models/model_groups.json"
CLIENT = ROOT / "src/gifr/baselines/models.py"
SUMMARIZER = ROOT / "scripts/summarize_run.py"

NEW_KEY = "gemini_3_6_flash"
OLD_KEYS = {"gemini_2_5_pro", "gemini_2_5_flash"}

NEW_CONFIG = {
    "provider": "openai_compatible",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "base_url_env": "GEMINI_BASE_URL",
    "api_key_env": "GEMINI_API_KEY",
    "model": "gemini-3.6-flash",
    # Gemini 3.6 Flash deprecates sampling parameters; null means do not send temperature.
    "temperature": None,
    "max_tokens": 512,
    "timeout_seconds": 180,
    "extra_body": {"reasoning_effort": "low"},
    "size_label": "api_flash",
    "parameter_count": "undisclosed",
    "notes": "Stable Gemini 3.6 Flash via Google's OpenAI-compatible endpoint; low reasoning effort.",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_models_json() -> None:
    root = load(MODELS)
    models = root.setdefault("models", {})
    models[NEW_KEY] = NEW_CONFIG
    # Keep old configs for reproducibility, but mark them legacy and remove them from groups.
    for key in OLD_KEYS:
        if key in models and isinstance(models[key], dict):
            note = str(models[key].get("notes", ""))
            if "Legacy" not in note:
                models[key]["notes"] = ("Legacy config; not included in benchmark groups. " + note).strip()
    save(MODELS, root)


def replace_group_members(members: list[str]) -> list[str]:
    result: list[str] = []
    inserted = False
    for member in members:
        if member in OLD_KEYS:
            if not inserted:
                result.append(NEW_KEY)
                inserted = True
            continue
        if member not in result:
            result.append(member)
    if not inserted and NEW_KEY not in result:
        result.append(NEW_KEY)
    return result


def patch_groups_json() -> None:
    root = load(GROUPS)
    groups = root.setdefault("groups", {})
    if "large" in groups:
        groups["large"]["members"] = replace_group_members(list(groups["large"].get("members", [])))
        groups["large"]["description"] = "Hosted API benchmark models, including GPT-4o and Gemini 3.6 Flash."
    groups["openai_google"] = {
        "description": "GPT-4o and Gemini 3.6 Flash",
        "members": ["gpt_4o", NEW_KEY],
    }
    save(GROUPS, root)


def patch_client() -> None:
    text = CLIENT.read_text(encoding="utf-8")

    if "temperature: float | None = 0.0" not in text:
        text = text.replace(
            "    temperature: float = 0.0\n",
            "    temperature: float | None = 0.0\n",
        )

    old_return = """        return cls(\n            key=key,"""
    if "raw_temperature = raw.get(\"temperature\", 0.0)" not in text:
        text = text.replace(
            old_return,
            """        raw_temperature = raw.get(\"temperature\", 0.0)\n        return cls(\n            key=key,""",
        )

    text = text.replace(
        "            temperature=float(raw.get(\"temperature\", 0.0)),\n",
        "            temperature=(None if raw_temperature is None else float(raw_temperature)),\n",
    )

    old_payload = """        payload: dict[str, Any] = {\n            \"model\": self.config.model,\n            \"temperature\": self.config.temperature,\n            \"max_tokens\": self.config.max_tokens,"""
    new_payload = """        payload: dict[str, Any] = {\n            \"model\": self.config.model,\n            \"max_tokens\": self.config.max_tokens,"""
    text = text.replace(old_payload, new_payload)

    marker = """        payload.update(self.config.extra_body)\n"""
    conditional = """        if self.config.temperature is not None:\n            payload[\"temperature\"] = self.config.temperature\n        payload.update(self.config.extra_body)\n"""
    if conditional not in text:
        text = text.replace(marker, conditional, 1)

    CLIENT.write_text(text, encoding="utf-8")


def patch_summarizer() -> None:
    if not SUMMARIZER.exists():
        return
    text = SUMMARIZER.read_text(encoding="utf-8")
    if '"gemini_3_6_flash": "Gemini-3.6-Flash"' not in text:
        anchor = '    "gpt_4o": "GPT-4o",\n'
        text = text.replace(anchor, anchor + '    "gemini_3_6_flash": "Gemini-3.6-Flash",\n')
    SUMMARIZER.write_text(text, encoding="utf-8")


def main() -> int:
    for path in (MODELS, GROUPS, CLIENT):
        if not path.exists():
            raise SystemExit(f"Missing required project file: {path}")
    patch_models_json()
    patch_groups_json()
    patch_client()
    patch_summarizer()
    print("Migrated benchmark Gemini model to gemini_3_6_flash / gemini-3.6-flash")
    print("Updated groups: large, openai_google; all inherits through @large")
    print("Gemini requests now omit temperature when config sets it to null")
    print("Test: ./run_models.sh gemini_3_6_flash 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
