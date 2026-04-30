from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .nodes import (
    ask_name_node,
    build_report,
    handle_experience_node,
    handle_intro_node,
    handle_name_node,
    handle_technical_node,
    make_technical_question,
)
from .state import InterviewState, build_initial_state

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - local fallback keeps the repo runnable without the package
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class ManualInterviewGraph:
    """Small fallback executor with the same high-level purpose as the LangGraph workflow."""

    def __init__(self, orchestrator: "InterviewOrchestrator") -> None:
        self._orchestrator = orchestrator

    async def ainvoke(self, state: InterviewState) -> InterviewState:
        return self._orchestrator._manual_step(state)


class InterviewOrchestrator:
    def __init__(
        self,
        *,
        session_id: str,
        config: dict[str, Any],
        storage_dir: Path,
        topic_openers: dict[str, str],
    ) -> None:
        self._config = config
        self._persona = config["persona"]
        self._topic_openers = topic_openers
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self.state = build_initial_state(session_id)
        self._graph = self._build_graph()

    @property
    def is_done(self) -> bool:
        return self.state["completed"]

    @property
    def session_path(self) -> Path:
        return self._storage_dir / f"{self.state['session_id']}.json"

    async def start(self) -> str:
        self.state = await self._graph.ainvoke(self.state)
        self._persist()
        return self.state["current_prompt"]

    async def handle_user_input(self, text: str) -> str:
        self.state["last_user_message"] = text
        self.state["transcript"].append({"role": "candidate", "text": text})
        self.state = await self._graph.ainvoke(self.state)
        if self.state["current_prompt"]:
            self.state["transcript"].append({"role": "agent", "text": self.state["current_prompt"]})
        self._persist()
        return self.state["current_prompt"]

    def _persist(self) -> None:
        self.session_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            return ManualInterviewGraph(self)

        graph = StateGraph(InterviewState)
        graph.add_node("ask_name", self._node_ask_name)
        graph.add_node("handle_name", self._node_handle_name)
        graph.add_node("handle_intro", self._node_handle_intro)
        graph.add_node("handle_experience", self._node_handle_experience)
        graph.add_node("handle_technical", self._node_handle_technical)
        graph.add_node("finalize", self._node_finalize)
        graph.add_conditional_edges(
            START,
            self._route_entry,
            {
                "ask_name": "ask_name",
                "handle_name": "handle_name",
                "handle_intro": "handle_intro",
                "handle_experience": "handle_experience",
                "handle_technical": "handle_technical",
                "finalize": "finalize",
            },
        )
        graph.add_edge("ask_name", END)
        graph.add_edge("handle_name", END)
        graph.add_edge("handle_intro", END)
        graph.add_edge("handle_experience", END)
        graph.add_edge("handle_technical", END)
        graph.add_edge("finalize", END)
        return graph.compile()

    def _route_entry(self, state: InterviewState) -> str:
        if not state["current_prompt"] and state["pending_question"] == "name":
            return "ask_name"

        pending = state["pending_question"]
        if pending == "name":
            return "handle_name"
        if pending == "intro":
            return "handle_intro"
        if pending.startswith("experience:"):
            return "handle_experience"
        if pending.startswith("technical:"):
            return "handle_technical"
        return "finalize"

    def _node_ask_name(self, state: InterviewState) -> InterviewState:
        return self._manual_step(state)

    def _node_handle_name(self, state: InterviewState) -> InterviewState:
        return self._manual_step(state)

    def _node_handle_intro(self, state: InterviewState) -> InterviewState:
        return self._manual_step(state)

    def _node_handle_experience(self, state: InterviewState) -> InterviewState:
        return self._manual_step(state)

    def _node_handle_technical(self, state: InterviewState) -> InterviewState:
        return self._manual_step(state)

    def _node_finalize(self, state: InterviewState) -> InterviewState:
        return self._manual_step(state)

    def _manual_step(self, state: InterviewState) -> InterviewState:
        next_state = deepcopy(state)
        pending = next_state["pending_question"]

        if not next_state["current_prompt"] and pending == "name":
            next_state.update(ask_name_node(next_state, self._persona))
            return next_state

        if pending == "name":
            next_state.update(handle_name_node(next_state, self._persona))
            return next_state

        if pending == "intro":
            next_state.update(handle_intro_node(next_state))
            if next_state["pending_question"].startswith("experience:"):
                first_topic = self._config["interview"]["experience_topics"][0]
                name = next_state["candidate_name"] or "there"
                next_state["current_prompt"] = f"Thanks, {name}. {self._topic_openers[first_topic]}"
            return next_state

        if pending.startswith("experience:"):
            next_state.update(
                handle_experience_node(
                    next_state,
                    self._config["interview"]["experience_topics"],
                    self._topic_openers,
                )
            )
            if next_state["pending_question"].startswith("technical:"):
                role = self._persona["interviewing_for"]
                next_state["current_prompt"] = make_technical_question(0, next_state, role)
            return next_state

        if pending.startswith("technical:"):
            num_questions = self._config["interview"]["technical"]["num_questions"]
            next_state.update(
                handle_technical_node(
                    next_state,
                    self._persona["interviewing_for"],
                    num_questions,
                )
            )
            if next_state["pending_question"].startswith("technical:"):
                idx = int(next_state["pending_question"].split(":")[1])
                next_state["current_prompt"] = make_technical_question(
                    idx,
                    next_state,
                    self._persona["interviewing_for"],
                )
            else:
                next_state["report_md"] = build_report(next_state, self._persona["interviewing_for"])
                name = next_state["candidate_name"] or "there"
                next_state["current_prompt"] = (
                    f"Thank you, {name}. That concludes the interview. "
                    "I've saved your interview summary and evaluation notes."
                )
            return next_state

        next_state["completed"] = True
        next_state["report_md"] = build_report(next_state, self._persona["interviewing_for"])
        return next_state
