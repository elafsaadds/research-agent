
"""Gradio web interface for the Research Agent."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from api import run_research_request


def as_dict(value):
    return value if isinstance(value, dict) else {}


def show_json(value):
    """Ensure JSON components always receive valid Python data."""
    if value is None:
        return {}

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
        from dataclasses import asdict
        return show_json(asdict(value))

    return {"value": str(value)}


def research_callback(question):
    """Run research and return report, plan, evidence, conflicts, diagnostics."""
    try:
        response = run_research_request(question)
        result = as_dict(response.get("result"))

        report_text = response.get("message")
        if not report_text:
            report_text = (
                "The research run returned no separate final message. "
                "Inspect the evidence and diagnostics tabs for details."
            )

        report = (
            "# Research Report\n\n"
            f"**Status:** {response.get('status', 'unknown')}  \n"
            f"**Model:** {response.get('model', 'unknown')}  \n\n"
            f"{report_text}"
        )

        plan = show_json(result.get("plan"))
        evidence = show_json(result.get("table"))
        conflicts = show_json(result.get("conflicts"))

        diagnostics = show_json({
            "question": response.get("question"),
            "status": response.get("status"),
            "model": response.get("model"),
            "answered": result.get("answered"),
            "total_sub_questions": result.get("total"),
            "replans": result.get("replans"),
            "original_question_answered": result.get("original_answered"),
            "model_calls": result.get("model_calls"),
            "findings_rejected": result.get("findings_rejected"),
            "latency_s": result.get("latency_s"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "search_log": result.get("search_log", []),
        })

        return report, plan, evidence, conflicts, diagnostics

    except Exception as exc:
        error_report = (
            "# Research run failed\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`\n\n"
            "Check the PowerShell terminal for details. "
            "Never share your API key or `.env` file."
        )
        error_data = {"error_type": type(exc).__name__, "error": str(exc)}
        return error_report, error_data, error_data, error_data, error_data


with gr.Blocks(title="Research Agent") as demo:
    gr.Markdown(
        """
        # Research Agent
        ### Evidence-grounded, multi-step research

        Enter a research question to run the existing research pipeline.
        Results appear below when the run finishes.
        """
    )

    question = gr.Textbox(
        label="Research question",
        placeholder="What are the applications of AI in healthcare?",
        lines=3,
    )

    run_button = gr.Button("Start Research", variant="primary")
    report_output = gr.Markdown()

    with gr.Tab("Research plan"):
        plan_output = gr.JSON(label="Plan")

    with gr.Tab("Evidence"):
        evidence_output = gr.JSON(label="Evidence table")

    with gr.Tab("Conflicts"):
        conflicts_output = gr.JSON(label="Detected conflicts")

    with gr.Tab("Run diagnostics"):
        diagnostics_output = gr.JSON(label="Run information")

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
