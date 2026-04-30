from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import EXPERIENCE_TOPIC_OPENERS
from graph import InterviewOrchestrator
from graph.nodes import extract_name, is_substantive_intro, score_answer


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_extract_name_handles_simple_name():
    assert extract_name("dakshi") == "Dakshi"
    assert extract_name("my name is dakshi goel") == "Dakshi Goel"


def test_intro_threshold():
    assert not is_substantive_intro("student")
    assert is_substantive_intro("I am doing my masters in data science and building AI products.")


def test_score_answer_returns_valid_range():
    result = score_answer("I would monitor latency, evaluate accuracy, and roll back if needed.")
    assert 1 <= int(result["score"]) <= 5
    assert result["reason"]


@pytest.mark.asyncio
async def test_orchestrator_happy_path(tmp_path: Path):
    orchestrator = InterviewOrchestrator(
        session_id="session-1",
        config=_cfg(),
        storage_dir=tmp_path,
        topic_openers=EXPERIENCE_TOPIC_OPENERS,
    )

    opening = await orchestrator.start()
    assert "tell me your name" in opening.lower()
    assert orchestrator.session_path.exists()

    prompt = await orchestrator.handle_user_input("My name is Dakshi")
    assert "dakshi" in prompt.lower()
    assert "background" in prompt.lower()

    prompt = await orchestrator.handle_user_input(
        "I am doing my masters in data science and I have one year of experience building AI products."
    )
    assert "walk me through your most recent role" in prompt.lower()

    prompt = await orchestrator.handle_user_input("I built a RAG pipeline for a real estate assistant.")
    assert "project you're proud of" in prompt.lower()

    prompt = await orchestrator.handle_user_input("I built a multi-agent finance platform and designed the orchestration pipeline.")
    assert "technical challenge" in prompt.lower()

    prompt = await orchestrator.handle_user_input("The context window was too small, so I reduced prompt size and changed the model.")
    assert "evaluate an ai feature" in prompt.lower()
    assert orchestrator.state["experience_summary"]

    prompt = await orchestrator.handle_user_input(
        "I would define evaluation datasets, measure quality and latency, and review failure cases before rollout."
    )
    assert "production ai assistant" in prompt.lower()

    prompt = await orchestrator.handle_user_input(
        "I would compare recent changes, inspect prompts and retrieval, check metrics, and roll back if risk was high."
    )
    assert "concludes the interview" in prompt.lower()
    assert orchestrator.is_done
    assert orchestrator.state["report_md"]
    assert len(orchestrator.state["scores"]) == 2

    saved = orchestrator.session_path.read_text(encoding="utf-8")
    assert "Dakshi" in saved
    assert "report_md" in saved
