"""Gradio web interface for the Research Agent."""

import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import gradio as gr

from app.api import run_research_request


def as_dict(value: Any) -> dict:
    """Return value as a dictionary when possible."""
    return value if isinstance(value, dict) else {}


def show_json(value: Any) -> Any:
    """Convert project objects into data accepted by Gradio JSON components."""
    if value is None:
        return {}

    # Convert the custom EvidenceTable object explicitly.
    if (
        hasattr(value, "rows")
        and hasattr(value, "rejected")
        and value.__class__.__name__ == "EvidenceTable"
    ):
        return {
            "rows": [
                {
                    "sub_question_id": row.sub_question_id,
                    "finding": show_json(row.finding),
                }
                for row in value.rows
            ],
            "rejected": [
                {
                    "sub_question_id": item.sub_question_id,
                    "finding": show_json(item.finding),
                    "reason": item.reason,
                }
                for item in value.rejected
            ],
        }

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {"value": value}

    if isinstance(value, (dict, list, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        return show_json(value.model_dump(mode="json"))

    if hasattr(value, "__dataclass_fields__"):
        return show_json(asdict(value))

    return {"value": str(value)}


def build_report(response: dict, result: dict) -> str:
    """Build the main Markdown research report."""
    report_text = response.get("message")

    if not report_text:
        report_text = (
            "The research run returned no separate final message. "
            "Inspect the evidence and diagnostics tabs for details."
        )

    return (
        "# Research Report\n\n"
        f"**Status:** {response.get('status', 'unknown')}  \n"
        f"**Model:** {response.get('model', 'unknown')}  \n\n"
        f"{report_text}"
    )


def build_diagnostics(response: dict, result: dict) -> dict:
    """Build structured run diagnostics for the UI."""
    return show_json(
        {
            "question": response.get("question"),
            "status": response.get("status"),
            "model": response.get("model"),
            "answered": result.get("answered"),
            "total_sub_questions": result.get("total"),
            "replans": result.get("replans"),
            "original_question_answered": result.get(
                "original_answered"
            ),
            "model_calls": result.get("model_calls"),
            "findings_rejected": result.get(
                "findings_rejected"
            ),
            "latency_s": result.get("latency_s"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get(
                "completion_tokens"
            ),
            "search_log": result.get("search_log", []),
        }
    )


def build_error_response(
    title: str,
    error_type: str,
    error_message: str,
) -> tuple[str, dict, dict, dict, dict]:
    """Build a consistent error response for all UI outputs."""
    error_report = (
        f"# {title}\n\n"
        f"**Error:** `{error_type}: {error_message}`\n\n"
        "Check the PowerShell terminal for additional "
        "diagnostic details. Never share your API key or `.env` file."
    )

    error_data = {
        "error_type": error_type,
        "error": error_message,
    }

    return (
        error_report,
        error_data,
        error_data,
        error_data,
        error_data,
    )


def research_callback(
    question: str,
) -> tuple[str, Any, Any, Any, Any]:
    """
    Run the research pipeline and return:

    1. Research report
    2. Research plan
    3. Evidence table
    4. Detected conflicts
    5. Run diagnostics
    """
    try:
        response = run_research_request(question)
        result = as_dict(response.get("result"))

        report = build_report(response, result)
        plan = show_json(result.get("plan"))
        evidence = show_json(result.get("table"))
        conflicts = show_json(result.get("conflicts"))
        diagnostics = build_diagnostics(response, result)

        return (
            report,
            plan,
            evidence,
            conflicts,
            diagnostics,
        )

    except ValueError as exc:
        return build_error_response(
            title="Research request rejected",
            error_type="ValidationError",
            error_message=str(exc),
        )

    except Exception as exc:
        return build_error_response(
            title="Research run failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


with gr.Blocks(title="Research Agent") as demo:
    gr.Markdown(
        """
# Research Agent

### Evidence-grounded, multi-step research

Enter a research question and run the existing
evidence-grounded research pipeline.

The interface exposes:

- Research report
- Research plan
- Evidence table
- Detected conflicts
- Run diagnostics
"""
    )

    question = gr.Textbox(
        label="Research question",
        placeholder=(
            "What are the applications of AI in healthcare?"
        ),
        lines=3,
    )

    run_button = gr.Button(
        "Start Research",
        variant="primary",
    )

    report_output = gr.Markdown()

    with gr.Tab("Research plan"):
        plan_output = gr.JSON(
            label="Plan"
        )

    with gr.Tab("Evidence"):
        evidence_output = gr.JSON(
            label="Evidence table"
        )

    with gr.Tab("Conflicts"):
        conflicts_output = gr.JSON(
            label="Detected conflicts"
        )

    with gr.Tab("Run diagnostics"):
        diagnostics_output = gr.JSON(
            label="Run information"
        )

    outputs = [
        report_output,
        plan_output,
        evidence_output,
        conflicts_output,
        diagnostics_output,
    ]

    run_button.click(
        fn=research_callback,
        inputs=question,
        outputs=outputs,
    )

    question.submit(
        fn=research_callback,
        inputs=question,
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
