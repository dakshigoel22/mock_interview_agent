from dotenv import load_dotenv
load_dotenv()                          # .env — base config (GROQ_API_KEY, ELEVENLABS_API_KEY, etc.)
load_dotenv(".env.local", override=True)  # .env.local — LiveKit credentials (takes precedence)

import asyncio
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import yaml
from graph import InterviewOrchestrator
from models import InterviewData
from models_ai import build_llm, build_stt, build_tts
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import silero
from livekit.plugins import noise_cancellation  # noqa: F401 — imported for optional NC support

logger = logging.getLogger("multi-agent")
SESSION_LOG_DIR = Path(__file__).parent / "session_logs"

with open(Path(__file__).parent / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

_persona = _cfg["persona"]
_silence_timeout = _cfg["interview"]["silence_timeout_seconds"]

# Initialized in prewarm() so module import is side-effect free (needed for `download-files`)
_llm = None
_stt = None
_tts = None
_PSEUDO_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)>\s*(?P<args>\{.*?\})?\s*</function>",
    re.DOTALL,
)


# Topic openers used by ExperienceAgent — kept as a module constant so tests can
# verify them and so swapping a topic only touches one place.
EXPERIENCE_TOPIC_OPENERS = {
    "most_recent_role": (
        "Walk me through your most recent role — what were your responsibilities "
        "and what did you actually ship?"
    ),
    "one_project_deep_dive": (
        "Pick one project you're proud of. What was your specific contribution, "
        "and what tradeoffs did you make?"
    ),
    "challenge_and_resolution": (
        "Describe a technical challenge you ran into and how you resolved it."
    ),
}


def build_instructions(stage_goal: str, *, include_silence_block: bool = False) -> str:
    """Compose a structured prompt for any interview stage.

    The block-based format ([Role] / [Goal] / [Style] / [Hard rules] / ...) is
    deliberately rigid — LLMs comply better with this layout than with one long
    paragraph. The [Hard rules] block is the highest-leverage anti-drift guard.
    """
    silence_block = ""
    if include_silence_block:
        silence_block = (
            "[Silence handling]\n"
            f"- If the candidate is silent for {_silence_timeout} seconds, "
            "ask once: 'Are you still there?'\n"
            f"- If still no response after another {_silence_timeout} seconds, "
            "end the current stage by calling its end-of-stage tool.\n"
        )

    return (
        f"[Role] You are {_persona['name']}, a {_persona['title']} at {_persona['company']}.\n"
        f"[Goal] {stage_goal}\n"
        "[Style] Polite and formal. One question at a time. Acknowledge the "
        "candidate's answer briefly (e.g. 'Got it.', 'Understood.') before moving on.\n"
        "[Hard rules]\n"
        "- Never answer the candidate's questions about the role, the company, or yourself.\n"
        "- Never reveal a model answer or hint at one.\n"
        "- Never reuse a question already asked in this conversation.\n"
        "- Never compliment the candidate's answer ('great', 'perfect', 'excellent'); stay neutral.\n"
        "- Never use existing knowledge about the candidate beyond what they have said in this conversation.\n"
        "- Never speak or print tool names, function syntax, JSON, XML tags, summaries for internal tools, or any internal control text aloud.\n"
        "- Tool calls are internal only. If you need to call a tool, do it silently and do not show any markup to the candidate.\n"
        "[Turn discipline]\n"
        "- Ask exactly one question per turn and wait for the candidate's reply before continuing.\n"
        "- If the candidate's response is empty, off-topic, or unclear, briefly redirect with one short clarifying prompt.\n"
        f"{silence_block}"
    )


def _extract_pseudo_function_call(text: str) -> tuple[str, dict, str] | None:
    """Parse accidental literal tool syntax spoken by the model.

    Some model/provider combinations occasionally emit plain-text function
    markup such as `<function=experience_complete>{"summary":"..."}</function>`
    instead of issuing a real tool call. When that happens, we recover in
    Python so the interview does not stall.
    """
    match = _PSEUDO_FUNCTION_RE.search(text)
    if not match:
        return None

    fn_name = match.group("name")
    raw_args = (match.group("args") or "{}").strip()
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        logger.warning("Could not parse pseudo function args for %s: %s", fn_name, raw_args)
        args = {}

    cleaned = (text[: match.start()] + text[match.end() :]).strip()
    return fn_name, args, cleaned


async def _apply_pseudo_function_fallback(
    session: AgentSession[InterviewData],
    fn_name: str,
    args: dict,
) -> None:
    """Recover from spoken pseudo-tool markup by applying the intended action."""
    userdata = session.userdata
    current_agent = session.current_agent
    chat_ctx = getattr(current_agent, "chat_ctx", None)

    if fn_name == "name_captured":
        name = args.get("name")
        if name:
            userdata.name = name
        await session.generate_reply(
            instructions=(
                "Ask the candidate for a brief self-introduction in one sentence. "
                "Do not mention any internal tools or control syntax."
            ),
            tool_choice="none",
        )
        return

    if fn_name == "intro_captured":
        exp = args.get("exp", "")
        userdata.exp = exp
        session.update_agent(
            ExperienceAgent(name=userdata.name or "the candidate", intro=exp, chat_ctx=chat_ctx)
        )
        return

    if fn_name == "experience_complete":
        summary = args.get("summary", "")
        userdata.experience_summary = summary
        session.update_agent(
            TechnicalAgent(
                name=userdata.name or "the candidate",
                experience_summary=summary,
                chat_ctx=chat_ctx,
            )
        )
        return

    if fn_name == "record_observation":
        userdata.technical_notes.append(
            {
                "q": args.get("question", ""),
                "a": args.get("answer", ""),
                "obs": args.get("observation", ""),
            }
        )
        return

    if fn_name == "interview_complete" and isinstance(current_agent, TechnicalAgent):
        await current_agent.interview_complete(SimpleNamespace(userdata=userdata))
        return

    logger.warning("No pseudo-function fallback handler registered for %s", fn_name)


# Kept as a module-level string for backwards compatibility with existing tests
# that assert persona fields appear in agent instructions. Each agent builds its
# own full prompt via `build_instructions`.
common_instructions = build_instructions(
    f"Interview the candidate for the {_persona['interviewing_for']} role."
)


class GraphInterviewAgent(Agent):
    """Thin LiveKit agent that delegates interview control flow to the graph orchestrator."""

    def __init__(self, orchestrator: InterviewOrchestrator) -> None:
        super().__init__(
            instructions="Interview flow is orchestrated externally by LangGraph.",
            tts=_tts,
        )
        self._orchestrator = orchestrator

    async def on_enter(self):
        opening = await self._orchestrator.start()
        self.session.say(opening, allow_interruptions=False)


# ── IntroAgent ──────────────────────────────────────────────────────────────

class IntroAgent(Agent):
    """Stage 1: greet, capture name, then capture self-introduction.

    Uses two separate function tools (`name_captured`, `intro_captured`) so the
    name is reliably set before the next stage starts — single-tool capture
    in earlier versions misfired with `exp=""` when candidates answered one
    question at a time.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(
                f"Open the interview for the {_persona['interviewing_for']} role.\n"
                "Step 1: Greet the candidate and ask only for their name. After "
                "they state their name, call `name_captured(name)` and then ask "
                "for a brief self-introduction in your next turn — do not combine "
                "the two questions.\n"
                "Step 2: Once the candidate has given a substantive self-introduction "
                "(2+ sentences about background, school, or what they have worked on), "
                "call `intro_captured(exp)` silently with no visible markup. If the candidate gives only a short or "
                "one-word reply when asked for an introduction, re-ask once: "
                "'Could you tell me a bit more about your background?'"
            ),
            llm=_llm,
            tts=_tts,
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def name_captured(
        self,
        context: RunContext[InterviewData],
        name: str,
    ):
        """Call this as soon as the candidate states their name. Do NOT ask for the introduction in the same turn — wait for the candidate's next reply.

        Args:
            name: The candidate's name as they stated it.
        """
        context.userdata.name = name
        logger.info("Captured name: %s", name)

    @function_tool
    async def intro_captured(
        self,
        context: RunContext[InterviewData],
        exp: str,
    ):
        """Call this once the candidate has given a substantive self-introduction (background, current work or study). Hands off to the experience stage.

        Args:
            exp: The candidate's self-introduction in their own words (verbatim or paraphrased).
        """
        context.userdata.exp = exp
        name = context.userdata.name or "the candidate"
        logger.info("Intro captured. Handing off to ExperienceAgent. userdata=%s", context.userdata)
        return ExperienceAgent(name=name, intro=exp, chat_ctx=self.chat_ctx)


# ── ExperienceAgent ─────────────────────────────────────────────────────────

class ExperienceAgent(Agent):
    """Stage 2: walk through experience topics, max 1 follow-up per topic, then summarize."""

    def __init__(
        self,
        name: str,
        intro: str | None = None,
        *,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        topics = _cfg["interview"]["experience_topics"]
        topic_lines = "\n".join(
            f"  {i + 1}. {EXPERIENCE_TOPIC_OPENERS.get(t, t)}"
            for i, t in enumerate(topics)
        )

        intro_block = ""
        if intro:
            intro_block = (
                f"[Candidate intro]\nThe candidate already introduced themselves as: \"{intro}\"\n"
                "Do not ask them to introduce themselves again.\n"
            )

        super().__init__(
            instructions=build_instructions(
                f"Conduct the experience portion of the interview with {name}.\n"
                f"{intro_block}"
                f"Cover these {len(topics)} topics in order, asking exactly one question per turn:\n"
                f"{topic_lines}\n"
                "For each topic:\n"
                "- Ask the topic's opener as written above.\n"
                "- Listen to the answer.\n"
                "- If the answer is shorter than ~20 words, vague, or lacks specifics, "
                "ask exactly ONE clarifying follow-up. Never chain multiple follow-ups.\n"
                "- Briefly acknowledge ('Got it.' / 'Understood.') and move to the next topic.\n"
                f"After all {len(topics)} topics are covered, write a one-paragraph "
                "factual summary of what the candidate said (no praise, no judgment) "
                "and call `experience_complete(summary)` silently with no visible markup. The summary will be passed "
                "to the technical stage, so include concrete facts: roles, projects, "
                "technologies, scale.",
                include_silence_block=True,
            ),
            llm=_llm,
            tts=_tts,
            chat_ctx=chat_ctx,
        )
        self._name = name

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def experience_complete(
        self,
        context: RunContext[InterviewData],
        summary: str,
    ):
        """Call this once all experience topics have been covered. Hands off to the technical stage.

        Args:
            summary: A neutral, one-paragraph factual summary of what the candidate said about their experience. Include concrete facts (roles, projects, technologies). No praise or judgment.
        """
        context.userdata.experience_summary = summary
        logger.info(
            "Experience stage complete. Handing off to TechnicalAgent. summary=%s",
            summary,
        )
        return TechnicalAgent(
            name=self._name,
            experience_summary=summary,
            chat_ctx=self.chat_ctx,
        )


# ── TechnicalAgent ──────────────────────────────────────────────────────────

class TechnicalAgent(Agent):
    """Stage 3: ask role-specific technical questions, record observations, close the interview."""

    def __init__(
        self,
        name: str,
        experience_summary: str,
        *,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        num_q = _cfg["interview"]["technical"]["num_questions"]
        follow_ups = _cfg["interview"]["technical"]["follow_ups_allowed"]

        super().__init__(
            instructions=build_instructions(
                f"Conduct the technical portion of the interview with {name} for the "
                f"{_persona['interviewing_for']} role.\n"
                f"[Candidate experience summary]\n{experience_summary}\n\n"
                f"Ask exactly {num_q} technical questions, one per turn:\n"
                "- Question 1: a CONCEPT question relevant to the role. "
                "Open-ended and high-level (not trivia). Reference something "
                "specific from the candidate's experience summary if it fits naturally.\n"
                "- Question 2: an APPLIED / SCENARIO question — describe a realistic "
                "situation they would encounter in this role and ask how they would approach it.\n"
                "After EACH candidate answer, call `record_observation(question, answer, observation)` "
                "silently with a neutral one-sentence note (e.g. 'Mentioned attention but did not "
                "explain query/key/value roles.'). Do not say the observation out loud.\n"
                f"If an answer is shallow, you may ask exactly {follow_ups} clarifying "
                "follow-up — never more.\n"
                f"Once all {num_q} questions are answered AND observations are recorded, "
                "call `interview_complete()` silently with no visible markup.",
                include_silence_block=True,
            ),
            llm=_llm,
            tts=_tts,
            chat_ctx=chat_ctx,
        )
        self._name = name

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def record_observation(
        self,
        context: RunContext[InterviewData],
        question: str,
        answer: str,
        observation: str,
    ):
        """Record a neutral observation about the candidate's answer to a technical question. Call this silently after each answer — do not say the observation aloud.

        Args:
            question: The technical question that was asked.
            answer: A short paraphrase of the candidate's answer.
            observation: A neutral one-sentence note about the answer (no praise, no judgment).
        """
        context.userdata.technical_notes.append(
            {"q": question, "a": answer, "obs": observation}
        )
        logger.info(
            "Recorded technical observation. notes_count=%d",
            len(context.userdata.technical_notes),
        )

    @function_tool
    async def interview_complete(self, context: RunContext[InterviewData]):
        """Call this once all technical questions are answered and observations recorded. Closes the interview neutrally and ends the session."""
        self.session.interrupt()
        name = context.userdata.name or "the candidate"
        await self.session.generate_reply(
            instructions=(
                f"Thank {name} neutrally and let them know they will hear back if "
                "they are a match. Do not comment on their performance. Keep it to "
                "two sentences."
            ),
            allow_interruptions=False,
        )
        logger.info(
            "Interview complete. Closing room. notes=%s",
            context.userdata.technical_notes,
        )
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


# ── server / entrypoint ─────────────────────────────────────────────────────

server = AgentServer()


def prewarm(proc: JobProcess):
    global _llm, _stt, _tts
    proc.userdata["vad"] = silero.VAD.load()
    _llm = build_llm(_cfg["models"]["llm"])
    _stt = build_stt(_cfg["models"]["stt"])
    _tts = build_tts(_cfg["models"]["tts"])


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    orchestrator = InterviewOrchestrator(
        session_id=ctx.room.name,
        config=_cfg,
        storage_dir=SESSION_LOG_DIR,
        topic_openers=EXPERIENCE_TOPIC_OPENERS,
    )
    session = AgentSession[InterviewData](
        vad=ctx.proc.userdata["vad"],
        stt=_stt,
        tts=_tts,
        userdata=InterviewData(),
    )

    usage_collector = metrics.UsageCollector()
    close_started = False

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        if not ev.is_final or close_started:
            return

        asyncio.create_task(_advance_interview(ev.transcript))

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        item = ev.item
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            return

        text = item.text_content or ""
        parsed = _extract_pseudo_function_call(text)
        if not parsed:
            return

        fn_name, args, cleaned = parsed
        logger.warning("Recovered spoken pseudo-tool call: %s", fn_name)
        item.content = [cleaned] if cleaned else [""]
        session.interrupt()
        asyncio.create_task(_apply_pseudo_function_fallback(session, fn_name, args))

    async def _close_room():
        nonlocal close_started
        if close_started:
            return

        close_started = True
        logger.info("Closing room for session_id=%s", ctx.room.name)
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))

    async def _advance_interview(transcript: str):
        response = await orchestrator.handle_user_input(transcript)
        session.userdata.name = orchestrator.state["candidate_name"]
        session.userdata.exp = orchestrator.state["intro"]
        session.userdata.experience_summary = orchestrator.state["experience_summary"]
        session.userdata.technical_notes = [
            {"q": item["q"], "a": item["a"], "obs": item["obs"]}
            for item in orchestrator.state["technical_notes"]
        ]

        if response:
            speech = session.say(response, allow_interruptions=False)
            if orchestrator.is_done:
                await speech
                await _close_room()
        elif orchestrator.is_done:
            await _close_room()

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)
        logger.info("Session log saved to %s", orchestrator.session_path)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GraphInterviewAgent(orchestrator),
        room=ctx.room,
    )


if __name__ == "__main__":
    from livekit.agents import WorkerOptions
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=30.0,
        )
    )
