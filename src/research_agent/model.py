"""Model boundary: the rest of the agent only sees ModelClient."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
from dotenv import dotenv_values

_ENV_FILE = dotenv_values(".env")


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, falling back to .env."""

    value = os.getenv(name)

    if value is not None:
        return value

    value = _ENV_FILE.get(name)

    if value is not None:
        return str(value)

    return default


class ModelError(RuntimeError):
    """Raised when the model backend is unreachable or returns invalid data."""


@dataclass
class ModelResponse:
    """Standard response returned by every model backend."""

    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelClient(Protocol):
    """Common interface used by the Research Agent."""

    name: str

    def generate(self, messages: list[dict]) -> ModelResponse:
        ...


class MockModel:
    """Deterministic, free, offline model used for development and tests."""

    name = "mock"

    def generate(self, messages: list[dict]) -> ModelResponse:
        start = time.perf_counter()

        last_user = next(
            (
                message.get("content", "")
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

        text = f"[MOCK] Answer to: {last_user[:80]}"

        return ModelResponse(
            text=text,
            model="mock",
            latency_s=time.perf_counter() - start,
            prompt_tokens=sum(
                len(str(message.get("content", "")).split())
                for message in messages
            ),
            completion_tokens=len(text.split()),
        )


class OllamaModel:
    """Local model accessed through Ollama's /api/chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 4096,
        timeout_s: float = 300.0,
        response_schema: dict | None = None,
    ):
        self.model = model
        self.response_schema = response_schema

        self.name = f"ollama:{model}" + (
            "+schema" if response_schema else ""
        )

        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s

    def generate(self, messages: list[dict]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        if self.response_schema:
            payload["format"] = self.response_schema

        start = time.perf_counter()

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s,
            )

            response.raise_for_status()
            data = response.json()

        except httpx.HTTPError as exc:
            raise ModelError(
                f"Ollama request failed: {exc}"
            ) from exc

        try:
            text = data["message"]["content"]

        except (KeyError, TypeError) as exc:
            raise ModelError(
                f"Unexpected Ollama response: {data!r}"
            ) from exc

        return ModelResponse(
            text=text,
            model=self.model,
            latency_s=time.perf_counter() - start,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )


def _gemini_compatible_schema(value):
    """
    Convert the project's JSON schema into a Gemini-compatible schema.

    Gemini does not accept some standard JSON Schema keywords such as
    additionalProperties. The original project schema is not modified;
    only the copy sent to Gemini is cleaned.
    """

    if isinstance(value, dict):
        return {
            key: _gemini_compatible_schema(item)
            for key, item in value.items()
            if key != "additionalProperties"
        }

    if isinstance(value, list):
        return [
            _gemini_compatible_schema(item)
            for item in value
        ]

    return value


class GeminiModel:
    """Gemini model accessed through Google's generateContent REST API."""

    ENV_KEY_NAME = "GEMINI_API_KEY2"

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        response_schema: dict | None = None,
    ):
       
        self.model = model

        self.api_key = (
            api_key
            or os.getenv(self.ENV_KEY_NAME)
        )

        self.temperature = temperature
        self.timeout_s = timeout_s
        self.response_schema = response_schema

        self.name = f"gemini:{model}" + (
            "+schema" if response_schema else ""
        )

        if not self.api_key:
            raise ModelError(
                "GEMINI_API_KEY2 is not configured. "
                "Check the .env file."
            )

        self.base_url = (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models"
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:

        system_parts: list[str] = []
        contents: list[dict] = []

        for message in messages:

            role = message.get(
                "role",
                "user",
            )

            content = str(
                message.get(
                    "content",
                    "",
                )
            )

            if role == "system":
                system_parts.append(content)
                continue

            gemini_role = (
                "model"
                if role == "assistant"
                else "user"
            )

            contents.append(
                {
                    "role": gemini_role,
                    "parts": [
                        {
                            "text": content,
                        }
                    ],
                }
            )

        system_instruction = (
            "\n\n".join(system_parts)
            if system_parts
            else None
        )

        return system_instruction, contents

    def generate(
        self,
        messages: list[dict],
    ) -> ModelResponse:

        system_instruction, contents = (
            self._convert_messages(messages)
        )

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": system_instruction,
                    }
                ]
            }

        if self.response_schema:

            compatible_schema = (
                _gemini_compatible_schema(
                    self.response_schema
                )
            )

            payload["generationConfig"][
                "responseMimeType"
            ] = "application/json"

            payload["generationConfig"][
                "responseSchema"
            ] = compatible_schema

        url = (
            f"{self.base_url}/"
            f"{self.model}:generateContent"
        )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        start = time.perf_counter()

        try:

            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )

            if response.status_code >= 400:
                raise ModelError(
                    "Gemini request failed "
                    f"(HTTP {response.status_code}): "
                    f"{response.text[:1000]}"
                )

            data = response.json()

        except httpx.HTTPError as exc:
            raise ModelError(
                f"Gemini request failed: {exc}"
            ) from exc

        try:

            candidates = data["candidates"]

            if not candidates:
                raise ModelError(
                    f"Gemini returned no candidates: {data!r}"
                )

            content = candidates[0]["content"]

            parts = content["parts"]

            text = "".join(
                str(part.get("text", ""))
                for part in parts
            )

            if not text:
                raise ModelError(
                    f"Gemini returned empty text: {data!r}"
                )

        except (KeyError, TypeError, IndexError) as exc:
            raise ModelError(
                f"Unexpected Gemini response: {data!r}"
            ) from exc

        usage = data.get(
            "usageMetadata",
            {},
        )

        return ModelResponse(
            text=text,
            model=self.model,
            latency_s=time.perf_counter() - start,
            prompt_tokens=usage.get(
                "promptTokenCount"
            ),
            completion_tokens=usage.get(
                "candidatesTokenCount"
            ),
        )



def _decision_schema() -> dict | None:
    """Return the decision schema only when explicitly enabled."""

    enabled = os.getenv(
        "STRUCTURED_OUTPUT",
        "",
    ).strip().lower()

    if enabled not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    from research_agent.schemas import decision_json_schema

    return decision_json_schema()


def get_model() -> ModelClient:
    """Select the model backend from environment settings."""

    provider = (
        _env("MODEL_PROVIDER", "mock") or "mock"
    ).strip().lower()

    schema = _decision_schema()

    if provider == "mock":
        return MockModel()

    if provider == "ollama":
        return OllamaModel(
            model=_env(
                "OLLAMA_MODEL",
                "qwen2.5:3b",
            ) or "qwen2.5:3b",
            base_url=_env(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ) or "http://localhost:11434",
            response_schema=schema,
        )

    if provider == "gemini":
        api_key = _env(
            GeminiModel.ENV_KEY_NAME
        )

        if not api_key:
            raise ModelError(
                "GEMINI_API_KEY2 is not configured. "
                "Check the .env file."
            )

        return GeminiModel(
            model=_env(
                "GEMINI_MODEL",
                "gemini-3.1-flash-lite",
            ) or "gemini-3.1-flash-lite",
            api_key=api_key,
            response_schema=schema,
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER: {provider}"
    )

    schema = _decision_schema()

    if provider == "mock":
        return MockModel()

    if provider == "ollama":
        return OllamaModel(
            model=_env(
                "MODEL_NAME",
                "qwen2.5:3b",
            ) or "qwen2.5:3b",
            base_url=_env(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ) or "http://localhost:11434",
            response_schema=schema,
        )

    if provider == "gemini":

        api_key = _env(
            GeminiModel.ENV_KEY_NAME
        )

        if not api_key:
            raise ModelError(
                "GEMINI_API_KEY2 is not configured. "
                "Check the .env file."
            )

        return GeminiModel(
            model=_env(
                "MODEL_NAME",
                "gemini-3.1-flash-lite",
            ) or "gemini-3.1-flash-lite",
            api_key=api_key,
            response_schema=schema,
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER: {provider}"
    )
