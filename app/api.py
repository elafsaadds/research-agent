"""Application service for running the evidence-grounded research agent."""

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from research_agent.model import get_model
from research_agent.policy import Budget
from research_agent.research import run_research


MAX_QUESTION_LENGTH = 2000


def to_plain_data(value: Any) -> Any:
    """Convert project result objects into JSON-friendly Python values."""
    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        return to_plain_data(asdict(value))

    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump(mode="json"))

    if isinstance(value, dict):
        return {
            str(key): to_plain_data(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            to_plain_data(item)
            for item in value
        ]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def validate_question(question: Any) -> str:
    """Validate and normalize a research question."""
    if not isinstance(question, str):
        raise ValueError("Research question must be text.")

    question = question.strip()

    if not question:
        raise ValueError("Please enter a research question.")

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Research question is too long. "
            f"Maximum length is {MAX_QUESTION_LENGTH} characters."
        )

    return question


def build_response(question: str, result: Any, model: Any) -> dict:
    """Build the stable application response returned to the UI."""
    data = to_plain_data(result)

    if not isinstance(data, dict):
        data = {"result": data}

    return {
        "question": question,
        "model": getattr(
            result,
            "model",
            getattr(model, "name", "unknown"),
        ),
        "status": getattr(result, "status", "unknown"),
        "message": getattr(result, "message", None),
        "result": data,
    }


def run_research_request(question: str) -> dict:
    """Run the real research pipeline and return display-ready result data."""
    question = validate_question(question)

    model = get_model()
    budget = Budget.from_env()

    result = run_research(
        question=question,
        model=model,
        budget=budget,
    )

    return build_response(
        question=question,
        result=result,
        model=model,
    )