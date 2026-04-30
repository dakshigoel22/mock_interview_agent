from dotenv import load_dotenv
load_dotenv()                          # .env — base config (GROQ_API_KEY, HF_TOKEN, etc.)
load_dotenv(".env.local", override=True)  # .env.local — LiveKit credentials (takes precedence)

import logging
from pathlib import Path

import yaml
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

with open(Path(__file__).parent / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

_persona = _cfg["persona"]

# Initialized in prewarm() so module import is side-effect free (needed for `download-files`)
_llm = None
_stt = None
_tts = None

common_instructions = (
    f"You are {_persona['name']}, a {_persona['title']} at {_persona['company']}. "
    f"You are taking an interview for the candidate for the {_persona['interviewing_for']} role. "
    "You have to be polite and formal. Do not use any existing knowledge. "
    "Do not repeat what the user mentions. DO NOT ANSWER ANY USER QUESTION AS YOU ARE AN INTERVIEWER."
)


class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{common_instructions} Your goal is to start the interview. Greet the candidate. "
                "Ask the candidate for their name and a brief introduction."
            ),
            llm=_llm,
            tts=_tts,
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def information_gathered(
        self,
        context: RunContext[InterviewData],
        name: str,
        exp: str,
    ):
        """Called when the candidate has provided their name and introduction.

        Args:
            name: The full name of the candidate.
            exp: The candidate's self-introduction.
        """
        context.userdata.name = name
        context.userdata.exp = exp

        experience_agent = ExperienceAgent(name, chat_ctx=self.chat_ctx)
        logger.info("Switching to ExperienceAgent. userdata=%s", context.userdata)
        return experience_agent


class ExperienceAgent(Agent):
    """Asks the candidate about their previous work experience."""

    def __init__(self, name: str, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                f"{common_instructions} The candidate's name is {name}. "
                "Ask the candidate to describe their past work experiences, including any internships or full-time roles. "
                "Do not ask follow-up questions. "
                f"If the candidate does not respond for {_cfg['interview']['silence_timeout_seconds']} seconds, "
                f"verify by asking '{name}, are you still there?' "
                "If there is still no response, call `interview_finished`."
            ),
            llm=_llm,
            tts=_tts,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def interview_finished(self, context: RunContext[InterviewData], reason: str):
        """Call this when the experience section of the interview is complete.
        
        Args:
            reason: A short explanation of why the interview is finished.
        """
        self.session.interrupt()
        await self.session.generate_reply(
            instructions=(
                f"Thank {context.userdata.name} for their time and let them know "
                "that if they are a suitable match they will receive a callback."
            ),
            allow_interruptions=False,
        )
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


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
    session = AgentSession[InterviewData](
        vad=ctx.proc.userdata["vad"],
        llm=_llm,
        stt=_stt,
        tts=_tts,
        userdata=InterviewData(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=IntroAgent(),
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
