"""Model boundary: the rest of the agent only sees ModelClient."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
from dotenv import load_dotenv


class ModelError(RuntimeError):
    """Raised when the model backend is unreachable or returns garbage."""


@dataclass
class ModelResponse:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelClient(Protocol):
    name: str

    def generate(self, messages: list[dict]) -> ModelResponse: ...


class MockModel:
    """Deterministic, free, offline. Used for development and unit tests."""

    name = "mock"

    def generate(self, messages: list[dict]) -> ModelResponse:
        start = time.perf_counter()
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        text = f"[MOCK] Answer to: {last_user[:80]}"
        return ModelResponse(
            text=text,
            model="mock",
            latency_s=time.perf_counter() - start,
            prompt_tokens=sum(len(m["content"].split()) for m in messages),
            completion_tokens=len(text.split()),
        )


class OllamaModel:
    """Local model through Ollama's /api/chat.

    If response_schema is given, Ollama constrains the output to that JSON schema.
    The application still validates the result (structure is not the same as safety).
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, num_ctx: int = 4096,
                 timeout_s: float = 300.0, response_schema: dict | None = None):
        self.model = model
        self.response_schema = response_schema
        self.name = f"ollama:{model}" + ("+schema" if response_schema else "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s

    def generate(self, messages: list[dict]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.response_schema:
            payload["format"] = self.response_schema
        start = time.perf_counter()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload,
                           timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            raise ModelError(f"Ollama request failed: {exc}") from exc
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ModelError(f"Unexpected Ollama response: {data!r}") from exc
        return ModelResponse(
            text=text,
            model=self.model,
            latency_s=time.perf_counter() - start,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )


def get_model() -> ModelClient:
    """Pick the backend from .env so the agent code never changes."""
    load_dotenv()
    provider = os.getenv("MODEL_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockModel()
    if provider == "ollama":
        schema = None
        if os.getenv("STRUCTURED_OUTPUT", "0") == "1":
            from research_agent.schemas import decision_json_schema

            schema = decision_json_schema()
        return OllamaModel(
            model=os.getenv("MODEL_NAME", "qwen2.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            response_schema=schema,
        )
    raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")

