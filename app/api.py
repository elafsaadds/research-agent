"""Application service for running the evidence-grounded research agent."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from research_agent.model import get_model
from research_agent.policy import Budget
from research_agent.research import run_research


def to_plain_data(value: Any) -> Any:
    """Convert project result objects into JSON-friendly Python values."""
    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        return to_plain_data(asdict(value))

    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump(mode="json"))

    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_plain_data(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def run_research_request(question: str) -> dict:
    """Run the real research pipeline and return display-ready result data."""
    question = (question or "").strip()

    if not question:
        raise ValueError("Please enter a research question.")

    model = get_model()
    budget = Budget.from_env()

    result = run_research(question=question, model=model, budget=budget)
    data = to_plain_data(result)

    if not isinstance(data, dict):
        data = {"result": data}

    return {
        "question": question,
        "model": getattr(result, "model", getattr(model, "name", "unknown")),
        "status": getattr(result, "status", "unknown"),
        "message": getattr(result, "message", None),
        "result": data,
    }
