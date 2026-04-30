from __future__ import annotations

from statistics import mean
from typing import Any

from .state import InterviewState


def _word_count(text: str) -> int:
    return len([part for part in text.strip().split() if part])


def extract_name(text: str) -> str | None:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return None

    lowered = stripped.lower()
    for prefix in ("my name is ", "i am ", "i'm ", "this is "):
        if lowered.startswith(prefix):
            candidate = stripped[len(prefix) :].strip(" .,!?:;")
            return candidate.title() if candidate else None

    parts = stripped.split()
    if 1 <= len(parts) <= 4:
        return stripped.title()

    return None


def is_substantive_intro(text: str) -> bool:
    return _word_count(text) >= 8


def summarize_experience(state: InterviewState) -> str:
    parts: list[str] = []
    if state["candidate_name"]:
        parts.append(f'{state["candidate_name"]} introduced themselves as: {state["intro"] or ""}'.strip())

    for answer in state["experience_answers"]:
        topic = answer["topic"].replace("_", " ")
        parts.append(f'{topic}: {answer["answer"]}')

    summary = " ".join(part.strip() for part in parts if part.strip())
    return summary[:1200] if summary else "No experience summary captured."


def make_technical_question(index: int, state: InterviewState, role: str) -> str:
    name = state["candidate_name"] or "there"
    summary = state["experience_summary"] or state["intro"] or "your background"
    if index == 0:
        return (
            f"Thanks {name}. Based on {summary}, could you explain how you would evaluate "
            f"an AI feature before shipping it for a {role} role?"
        )

    return (
        f"{name}, imagine a production AI assistant starts giving inconsistent answers after "
        "a model or prompt change. How would you debug the issue, reduce risk, and stabilize it?"
    )


def make_observation(answer: str) -> str:
    words = _word_count(answer)
    lowered = answer.lower()
    if words < 8:
        return "Answer was very brief and lacked implementation detail."
    if any(keyword in lowered for keyword in ("tradeoff", "latency", "evaluation", "metrics", "debug")):
        return "Included concrete engineering considerations and decision criteria."
    return "Answered the question but with limited depth on validation, tradeoffs, or production handling."


def score_answer(answer: str) -> dict[str, str | int]:
    words = _word_count(answer)
    lowered = answer.lower()
    score = 2
    if words >= 10:
        score = 3
    if words >= 25:
        score = 4
    if any(keyword in lowered for keyword in ("tradeoff", "evaluation", "metrics", "latency", "monitor", "debug", "rollback")):
        score = min(5, score + 1)

    if score <= 2:
        reason = "Brief answer with limited technical detail."
    elif score == 3:
        reason = "Reasonable answer, but more concrete production details would strengthen it."
    elif score == 4:
        reason = "Strong answer with useful technical detail."
    else:
        reason = "Very strong answer with practical engineering tradeoffs and evaluation detail."

    return {"score": score, "reason": reason}


def build_report(state: InterviewState, role: str) -> str:
    scores = [int(item["score"]) for item in state["scores"] if "score" in item]
    overall = round(mean(scores), 2) if scores else 0.0

    lines = [
        f"# Interview Report: {state['candidate_name'] or 'Candidate'}",
        "",
        f"- Role: {role}",
        f"- Overall score: {overall}/5",
        "",
        "## Experience Summary",
        state["experience_summary"] or "Not captured.",
        "",
        "## Technical Review",
    ]

    for idx, answer in enumerate(state["technical_answers"], start=1):
        note = state["technical_notes"][idx - 1]["obs"] if idx - 1 < len(state["technical_notes"]) else ""
        score = state["scores"][idx - 1] if idx - 1 < len(state["scores"]) else {"score": "N/A", "reason": ""}
        lines.extend(
            [
                f"### Question {idx}",
                f"- Prompt: {answer['question']}",
                f"- Answer: {answer['answer']}",
                f"- Observation: {note}",
                f"- Score: {score['score']}",
                f"- Reason: {score['reason']}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def ask_name_node(state: InterviewState, persona: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "intro_name",
        "pending_question": "name",
        "current_prompt": (
            f"Hello, I'm {persona['name']}, a {persona['title']} at {persona['company']}. "
            "Could you please tell me your name?"
        ),
    }


def handle_name_node(state: InterviewState, persona: dict[str, Any]) -> dict[str, Any]:
    text = state["last_user_message"] or ""
    name = extract_name(text)
    if not name:
        return {
            "stage": "intro_name",
            "pending_question": "name",
            "current_prompt": "I want to make sure I address you correctly. Could you please tell me your name?",
        }

    return {
        "candidate_name": name,
        "stage": "intro_background",
        "pending_question": "intro",
        "current_prompt": (
            f"Thanks, {name}. Could you tell me a bit about your background, what you're studying or working on, "
            "and the kind of AI work you've done so far?"
        ),
    }


def handle_intro_node(state: InterviewState) -> dict[str, Any]:
    text = state["last_user_message"] or ""
    name = state["candidate_name"] or "there"
    if not is_substantive_intro(text):
        return {
            "stage": "intro_background",
            "pending_question": "intro",
            "current_prompt": (
                f"Thanks, {name}. Could you tell me a bit more about your background, including what you've built or worked on?"
            ),
        }

    return {
        "intro": text,
        "stage": "experience",
        "pending_question": "experience:0",
    }


def handle_experience_node(
    state: InterviewState,
    experience_topics: list[str],
    topic_openers: dict[str, str],
) -> dict[str, Any]:
    pending = state["pending_question"]
    idx = int(pending.split(":")[1])
    text = state["last_user_message"] or ""
    answers = list(state["experience_answers"])
    if idx < len(experience_topics):
        answers.append({"topic": experience_topics[idx], "answer": text})

    next_idx = idx + 1
    if next_idx < len(experience_topics):
        return {
            "experience_answers": answers,
            "stage": "experience",
            "pending_question": f"experience:{next_idx}",
            "current_prompt": topic_openers[experience_topics[next_idx]],
        }

    summary = summarize_experience({**state, "experience_answers": answers})
    return {
        "experience_answers": answers,
        "experience_summary": summary,
        "stage": "technical",
        "pending_question": "technical:0",
    }


def handle_technical_node(state: InterviewState, role: str, num_questions: int) -> dict[str, Any]:
    pending = state["pending_question"]
    idx = int(pending.split(":")[1])
    text = state["last_user_message"] or ""
    current_question = state["current_prompt"]

    technical_answers = list(state["technical_answers"])
    technical_answers.append({"question": current_question, "answer": text})

    note = make_observation(text)
    notes = list(state["technical_notes"])
    notes.append({"q": current_question, "a": text, "obs": note})

    score = score_answer(text)
    scores = list(state["scores"])
    scores.append({"question": current_question, "score": score["score"], "reason": score["reason"]})

    next_idx = idx + 1
    if next_idx < num_questions:
        return {
            "technical_answers": technical_answers,
            "technical_notes": notes,
            "scores": scores,
            "stage": "technical",
            "pending_question": f"technical:{next_idx}",
        }

    return {
        "technical_answers": technical_answers,
        "technical_notes": notes,
        "scores": scores,
        "stage": "done",
        "pending_question": "done",
        "completed": True,
    }
