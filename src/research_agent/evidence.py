"""Evidence table: claims linked to sources, verified in code.

The model proposes findings. Code accepts a finding only if its source was retrieved
for that sub-question and its evidence quote really appears in the source text.
What code cannot check: that the claim is a faithful reading of the quote, or that
the quote answers the sub-question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from research_agent.policy import CITATION
from research_agent.schemas import Finding, ResearchPlan

MIN_EVIDENCE_CHARS = 25
MAX_EVIDENCE_CHARS = 500


def normalize(text: str) -> str:
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return " ".join(text.casefold().split())


def verify_finding(f: Finding, sources: dict[str, str]) -> str | None:
    """Return None when the finding is verified, else the reason it was rejected."""
    text = sources.get(f.source_id)
    if text is None:
        return f"source {f.source_id!r} was not retrieved for this sub-question"
    quote = normalize(f.evidence)
    if len(quote) < MIN_EVIDENCE_CHARS:
        return "evidence quote is too short to verify"
    if len(quote) > MAX_EVIDENCE_CHARS:
        return "evidence quote is too long"
    if quote not in normalize(text):
        return "evidence quote does not appear in the source text"
    return None


@dataclass
class Row:
    sub_question_id: str
    finding: Finding


@dataclass
class Rejected:
    sub_question_id: str
    finding: Finding
    reason: str


class EvidenceTable:
    def __init__(self) -> None:
        self.rows: list[Row] = []
        self.rejected: list[Rejected] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add(self, sub_question_id: str, finding: Finding,
            sources: dict[str, str]) -> str | None:
        """Add a finding if it verifies. Returns the rejection reason, or None."""
        reason = verify_finding(finding, sources)
        if reason:
            self.rejected.append(Rejected(sub_question_id, finding, reason))
            return reason
        key = (sub_question_id, finding.source_id, normalize(finding.evidence))
        if key not in self._seen:
            self._seen.add(key)
            self.rows.append(Row(sub_question_id, finding))
        return None

    def for_sub_question(self, sub_question_id: str) -> list[Finding]:
        return [r.finding for r in self.rows if r.sub_question_id == sub_question_id]

    def source_ids(self, sub_question_id: str | None = None) -> set[str]:
        return {r.finding.source_id for r in self.rows
                if sub_question_id is None or r.sub_question_id == sub_question_id}

    def answered(self, sub_question_id: str) -> bool:
        """A sub-question counts as answered only with a supporting or contradicting finding."""
        return any(f.stance in ("supports", "contradicts")
                   for f in self.for_sub_question(sub_question_id))


class Conflict(BaseModel):
    sub_question_id: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=2)
    summary: str = Field(min_length=1, max_length=400)


def validate_conflict(c: Conflict, table: EvidenceTable) -> str | None:
    ids = set(c.source_ids)
    if len(ids) < 2:
        return "a conflict needs at least two distinct sources"
    missing = sorted(ids - table.source_ids(c.sub_question_id))
    if missing:
        return ("sources not in the evidence table for this sub-question: "
                + ", ".join(missing))
    return None


def detect_stance_conflicts(table: EvidenceTable) -> list[Conflict]:
    """Conflicts found from stance labels alone, with no model call."""
    sub_ids = list(dict.fromkeys(r.sub_question_id for r in table.rows))
    out: list[Conflict] = []
    for sid in sub_ids:
        findings = table.for_sub_question(sid)
        sup = {f.source_id for f in findings if f.stance == "supports"}
        con = {f.source_id for f in findings if f.stance == "contradicts"}
        ids = sup | con
        if sup and con and len(ids) >= 2:
            out.append(Conflict(
                sub_question_id=sid, source_ids=sorted(ids),
                summary=("Sources take opposing stances. Supporting: "
                         + ", ".join(sorted(sup)) + ". Contradicting: "
                         + ", ".join(sorted(con)) + ".")))
    return out


def _clean(text: str) -> str:
    text = " ".join(text.split())
    return text.replace("|", "/").replace("[", "(").replace("]", ")")


def render_report(question: str, plan: ResearchPlan, table: EvidenceTable,
                  conflicts: list[Conflict]) -> str:
    """Deterministic report built only from the verified evidence table."""
    lines = [f"Question: {_clean(question)}", ""]
    answered = 0
    for sq in plan.sub_questions:
        lines.append(f"## {sq.id}. {_clean(sq.question)}")
        if table.answered(sq.id):
            answered += 1
        else:
            lines.append("No verified evidence was found for this sub-question.")
        for f in table.for_sub_question(sq.id):
            lines.append(f"- ({f.stance}) {_clean(f.claim)} [{f.source_id}]")
        for c in conflicts:
            if c.sub_question_id == sq.id:
                cites = " ".join(f"[{i}]" for i in c.source_ids)
                lines.append(f"- CONFLICT: {_clean(c.summary)} {cites}")
        lines.append("")
    lines.append(f"Coverage: {answered}/{len(plan.sub_questions)} sub-questions have verified evidence.")
    lines.append("")
    lines.append("## Evidence table")
    lines.append("| sub-question | source | stance | quote |")
    lines.append("|---|---|---|---|")
    for r in table.rows:
        lines.append(f'| {r.sub_question_id} | {r.finding.source_id} | '
                     f'{r.finding.stance} | "{_clean(r.finding.evidence)}" |')
    return "\n".join(lines)


def check_report_citations(text: str, table: EvidenceTable) -> list[str]:
    """Cited ids that are not in the evidence table (empty list means all known)."""
    known = table.source_ids()
    cited = list(dict.fromkeys(CITATION.findall(text)))
    return [c for c in cited if c not in known]
