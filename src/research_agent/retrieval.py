"""Evidence text lookup: what the extraction step may quote from.

Findings are verified against this text, so the model must be shown exactly this text.
"""
from __future__ import annotations

from research_agent import tools
from research_agent.tools import SearchHit


def source_text(hit: SearchHit) -> str:
    """Full document body for corpus hits; the abstract snippet for arXiv hits."""
    if hit.origin == "corpus":
        for d in tools.get_corpus().docs:
            if d.doc_id == hit.source_id:
                return d.body
    return hit.snippet
