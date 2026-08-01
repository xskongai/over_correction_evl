#!/usr/bin/env python3
"""Safely merge GPT-4o and Gemini configs into an existing project checkout."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "configs/models/zero_shot_models.json"
GROUPS = ROOT / "configs/models/model_groups.json"

ADDITIONS = {
    "gpt_4o": {
        "provider": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-2024-11-20",
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout_seconds": 180,
        "size_label": "api_flagship",
        "parameter_count": "undisclosed",
        "notes": "Pinned GPT-4o snapshot for reproducible zero-shot evaluation."
    },
    "gemini_2_5_pro": {
        "provider": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "base_url_env": "GEMINI_BASE_URL",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-pro",
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout_seconds": 300,
        "extra_body": {"reasoning_effort": "low"},
        "size_label": "api_flagship",
        "parameter_count": "undisclosed",
        "notes": "Gemini 2.5 Pro via Google OpenAI-compatible endpoint; low reasoning effort."
    },
    "gemini_2_5_flash": {
        "provider": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "base_url_env": "GEMINI_BASE_URL",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout_seconds": 180,
        "extra_body": {"reasoning_effort": "none"},
        "size_label": "api_flash",
        "parameter_count": "undisclosed",
        "notes": "Optional lower-cost Gemini 2.5 Flash baseline with thinking disabled."
    },
}

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main() -> int:
    models_root = load(MODELS)
    models = models_root.setdefault("models", {})
    models.update(ADDITIONS)
    save(MODELS, models_root)

    groups_root = load(GROUPS)
    groups = groups_root.setdefault("groups", {})
    large = groups.setdefault("large", {"description": "Hosted API models", "members": []})
    members = large.setdefault("members", [])
    for key in ("gpt_4o", "gemini_2_5_pro"):
        if key not in members:
            members.append(key)
    groups["openai_google"] = {
        "description": "GPT-4o and Gemini 2.5 Pro",
        "members": ["gpt_4o", "gemini_2_5_pro"],
    }
    save(GROUPS, groups_root)

    env_example = ROOT / ".env.example"
    if env_example.exists():
        text = env_example.read_text(encoding="utf-8")
        additions = []
        if "OPENAI_API_KEY=" not in text:
            additions.append("OPENAI_API_KEY=")
        if "GEMINI_API_KEY=" not in text:
            additions.append("GEMINI_API_KEY=")
        if additions:
            text = text.rstrip() + "\n" + "\n".join(additions) + "\n"
            env_example.write_text(text, encoding="utf-8")

    summarizer = ROOT / "scripts/summarize_run.py"
    if summarizer.exists():
        text = summarizer.read_text(encoding="utf-8")
        anchor = '    "glm5_2_dashscope": "GLM-5.2",\n'
        additions = (
            '    "gpt_4o": "GPT-4o",\n'
            '    "gemini_2_5_pro": "Gemini-2.5-Pro",\n'
            '    "gemini_2_5_flash": "Gemini-2.5-Flash",\n'
        )
        if '"gpt_4o": "GPT-4o"' not in text and anchor in text:
            text = text.replace(anchor, anchor + additions)
            summarizer.write_text(text, encoding="utf-8")

    print("Added: gpt_4o, gemini_2_5_pro, gemini_2_5_flash")
    print("Updated groups: large, openai_google; all inherits them through @large")
    print("Add OPENAI_API_KEY and GEMINI_API_KEY to your .env before real calls.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
