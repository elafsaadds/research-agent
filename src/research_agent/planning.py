"""Research plan: validation, progress state, bounded re-planning, overall status."""
from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.evidence import EvidenceTable, normalize
from research_agent.policy import Budget
from research_agent.schemas import ResearchPlan, SubQuestion, TerminalStatus

MAX_SUB_QUESTIONS = 4
MIN_QUESTION_CHARS = 8


def estimate_tool_calls(plan: ResearchPlan, n_sources: int = 2) -> int:
    return len(plan.sub_questions) * n_sources


def validate_plan(plan: ResearchPlan, budget: Budget, n_sources: int = 2,
                  max_sub: int = MAX_SUB_QUESTIONS) -> str | None:
    """Check a proposed plan before any step runs. Returns the problem, or None."""
    if len(plan.sub_questions) > max_sub:
        return f"too many sub-questions ({len(plan.sub_questions)} > {max_sub})"
    seen: set[str] = set()
    for s in plan.sub_questions:
        q = normalize(s.question)
        if len(q) < MIN_QUESTION_CHARS:
            return f"sub-question {s.id} is too short"
        if q in seen:
            return f"sub-question {s.id} duplicates another sub-question"
        seen.add(q)
    needed = estimate_tool_calls(plan, n_sources)
    if needed > budget.max_tool_calls:
        return (f"plan needs about {needed} tool calls but the budget allows "
                f"{budget.max_tool_calls}")
    return None


def fallback_plan(question: str) -> ResearchPlan:
    """Used when the model proposes no valid plan: the question itself, as one step."""
    return ResearchPlan(goal=question,
                        sub_questions=[SubQuestion(id="s1", question=question)])


@dataclass
class PlanState:
    plan: ResearchPlan
    max_replans: int = 2
    replans: int = 0
    history: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for s in self.plan.sub_questions:
            self.history.setdefault(s.id, [normalize(s.question)])

    def get(self, sub_question_id: str) -> SubQuestion:
        for s in self.plan.sub_questions:
            if s.id == sub_question_id:
                return s
        raise KeyError(sub_question_id)

    def mark(self, sub_question_id: str, status: str) -> None:
        self.get(sub_question_id).status = status

    def pending(self) -> list[SubQuestion]:
        return [s for s in self.plan.sub_questions if s.status == "pending"]

    def try_replan(self, sub_question_id: str, new_question: str) -> str | None:
        """Reword a failed sub-question. Returns the refusal reason, or None if accepted."""
        if self.replans >= self.max_replans:
            return "re-plan budget exhausted"
        new = normalize(new_question)
        if len(new) < MIN_QUESTION_CHARS:
            return "replacement question is too short"
        if new in self.history[sub_question_id]:
            return "replacement repeats an earlier question for this sub-question"
        sq = self.get(sub_question_id)
        sq.question = new_question.strip()
        sq.status = "pending"
        self.history[sub_question_id].append(new)
        self.replans += 1
        return None


def overall_status(state: PlanState, table: EvidenceTable) -> TerminalStatus:
    total = len(state.plan.sub_questions)
    answered = sum(1 for s in state.plan.sub_questions if table.answered(s.id))
    if answered == 0:
        return TerminalStatus.INSUFFICIENT_EVIDENCE
    if answered == total:
        return TerminalStatus.COMPLETED
    return TerminalStatus.PARTIAL
