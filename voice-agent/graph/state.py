from __future__ import annotations

from typing import Literal, TypedDict


Stage = Literal["intro_name", "intro_background", "experience", "technical", "done"]


class InterviewState(TypedDict):
    session_id: str
    stage: Stage
    candidate_name: str | None
    intro: str | None
    transcript: list[dict[str, str]]
    current_prompt: str
    pending_question: str
    experience_answers: list[dict[str, str]]
    technical_answers: list[dict[str, str]]
    experience_summary: str | None
    technical_notes: list[dict[str, str]]
    scores: list[dict[str, str | int]]
    report_md: str | None
    completed: bool
    last_user_message: str | None


def build_initial_state(session_id: str) -> InterviewState:
    return InterviewState(
        session_id=session_id,
        stage="intro_name",
        candidate_name=None,
        intro=None,
        transcript=[],
        current_prompt="",
        pending_question="name",
        experience_answers=[],
        technical_answers=[],
        experience_summary=None,
        technical_notes=[],
        scores=[],
        report_md=None,
        completed=False,
        last_user_message=None,
    )
