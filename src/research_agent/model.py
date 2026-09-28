import os
import time
import requests


class ModelResponse:
    def __init__(self, text, latency_s, completion_tokens):
        self.text = text
        self.latency_s = latency_s
        self.completion_tokens = completion_tokens


class OllamaModel:
    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:1.5b"
        )

    def generate(self, messages):
        start = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        latency = time.perf_counter() - start

        text = data["message"]["content"]

        completion_tokens = data.get("eval_count", 0)

        return ModelResponse(
            text=text,
            latency_s=latency,
            completion_tokens=completion_tokens
        )


def get_model():
    return OllamaModel()