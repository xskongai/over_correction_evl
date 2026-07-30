"""模型无关的 OpenAI-compatible 调用层。

当前 DeepSeek、OpenAI、Qwen、GLM，以及 Ollama / vLLM / LM Studio
暴露的本地接口都可以通过同一客户端接入。模型差异只放在 JSON 配置里，
实验脚本不包含 provider-specific 分支。
"""
from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ModelConfigError(ValueError):
    """模型配置无效。"""


class ModelRequestError(RuntimeError):
    """模型请求最终失败。"""


@dataclass(frozen=True)
class ModelConfig:
    key: str
    provider: str
    base_url: str
    model: str
    api_key_env: str = ""
    base_url_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    size_label: str = "unknown"
    parameter_count: str = "unknown"
    notes: str = ""

    @classmethod
    def from_dict(cls, key: str, raw: dict[str, Any]) -> "ModelConfig":
        provider = str(raw.get("provider", "openai_compatible"))
        if provider != "openai_compatible":
            raise ModelConfigError(
                f"模型 {key!r} 的 provider={provider!r} 暂不支持；"
                "当前请使用 openai_compatible"
            )
        required = [name for name in ("base_url", "model") if not raw.get(name)]
        if required:
            raise ModelConfigError(f"模型 {key!r} 缺少字段: {', '.join(required)}")
        base_url_env = str(raw.get("base_url_env", ""))
        configured_base_url = str(raw["base_url"])
        resolved_base_url = (
            os.environ.get(base_url_env, configured_base_url)
            if base_url_env
            else configured_base_url
        )
        if not resolved_base_url.strip():
            raise ModelConfigError(
                f"模型 {key!r} 的 base_url 为空；请设置 {base_url_env or 'base_url'}"
            )
        return cls(
            key=key,
            provider=provider,
            base_url=resolved_base_url.rstrip("/"),
            model=str(raw["model"]),
            api_key_env=str(raw.get("api_key_env", "")),
            base_url_env=base_url_env,
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 512)),
            timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
            extra_body=dict(raw.get("extra_body") or {}),
            extra_headers={str(k): str(v) for k, v in (raw.get("extra_headers") or {}).items()},
            size_label=str(raw.get("size_label", "unknown")),
            parameter_count=str(raw.get("parameter_count", "unknown")),
            notes=str(raw.get("notes", "")),
        )

    def public_dict(self) -> dict[str, Any]:
        """可安全写入实验元数据的配置，不包含 API key。"""
        return {
            "key": self.key,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "extra_body": self.extra_body,
            "extra_headers": self.extra_headers,
            "size_label": self.size_label,
            "parameter_count": self.parameter_count,
            "notes": self.notes,
        }


def load_model_config(path: Path, key: str) -> ModelConfig:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelConfigError(f"模型配置文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelConfigError(f"模型配置 JSON 无效: {path}: {exc}") from exc
    models = root.get("models") if isinstance(root, dict) else None
    if not isinstance(models, dict):
        raise ModelConfigError(f"{path} 顶层必须包含 models 对象")
    if key not in models:
        choices = ", ".join(sorted(models))
        raise ModelConfigError(f"未知 model-key {key!r}；可选: {choices}")
    raw = models[key]
    if not isinstance(raw, dict):
        raise ModelConfigError(f"模型 {key!r} 的配置必须是对象")
    return ModelConfig.from_dict(key, raw)


def list_model_keys(path: Path) -> list[str]:
    root = json.loads(path.read_text(encoding="utf-8"))
    models = root.get("models", {})
    return sorted(models) if isinstance(models, dict) else []


def load_dotenv(path: Path | None, *, override: bool = False) -> None:
    """最小 .env 读取器，避免为了一个变量增加运行时依赖。"""
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value


@dataclass(frozen=True)
class Completion:
    content: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str
    request_id: str
    attempts: int
    reasoning_chars: int = 0
    response_model: str = ""
    system_fingerprint: str = ""


class OpenAICompatibleClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        max_retries: int = 4,
        retry_base_seconds: float = 1.5,
    ) -> None:
        self.config = config
        self.max_retries = max(0, int(max_retries))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        if config.api_key_env and not self.api_key:
            raise ModelConfigError(
                f"环境变量 {config.api_key_env} 未设置。"
                f"请在 .env 中加入 {config.api_key_env}=你的APIKey"
            )

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def complete(self, prompt: str) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            # 为保证 zero-shot 条件不被额外 system prompt 改变，完整提示词只作为一条 user 消息。
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(self.config.extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"content-type": "application/json", **self.config.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_retries + 2):
            req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    req, timeout=self.config.timeout_seconds
                ) as response:
                    raw = response.read().decode("utf-8")
                    data = json.loads(raw)
                    choice = data["choices"][0]
                    message = choice.get("message") or {}
                    usage = data.get("usage") or {}
                    return Completion(
                        content=str(message.get("content") or ""),
                        latency_seconds=time.perf_counter() - started,
                        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                        completion_tokens=_optional_int(usage.get("completion_tokens")),
                        total_tokens=_optional_int(usage.get("total_tokens")),
                        finish_reason=str(choice.get("finish_reason") or ""),
                        request_id=str(
                            response.headers.get("x-request-id")
                            or data.get("id")
                            or ""
                        ),
                        attempts=attempt,
                        # 不保存模型推理正文，只记录长度用于诊断是否启用了思考模式。
                        reasoning_chars=len(str(message.get("reasoning_content") or "")),
                        response_model=str(data.get("model") or ""),
                        system_fingerprint=str(data.get("system_fingerprint") or ""),
                    )
            except urllib.error.HTTPError as exc:
                detail = _read_http_error(exc)
                last_exc = ModelRequestError(
                    f"HTTP {exc.code} from {self.endpoint}: {detail[:800]}"
                )
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
                last_exc = exc
            except (KeyError, IndexError, TypeError) as exc:
                last_exc = ModelRequestError(f"API 响应结构异常: {exc}")
                break

            if attempt <= self.max_retries:
                delay = self.retry_base_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0.0, min(1.0, delay * 0.2))
                time.sleep(delay)

        elapsed = time.perf_counter() - started
        raise ModelRequestError(
            f"模型调用失败（{self.config.key}，{elapsed:.1f}s，"
            f"尝试 {self.max_retries + 1} 次）: {last_exc}"
        ) from last_exc


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return str(exc)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
