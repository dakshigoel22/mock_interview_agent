"""Tests for IntroAgent, ExperienceAgent, and TechnicalAgent."""
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_livekit_stubs():
    """Build minimal stubs so agent.py can be imported without LiveKit installed."""
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = MagicMock()
    sys.modules.setdefault("dotenv", dotenv_mod)

    livekit = types.ModuleType("livekit")
    livekit_api = types.ModuleType("livekit.api")
    livekit_api.DeleteRoomRequest = MagicMock()
    livekit.api = livekit_api

    agents = types.ModuleType("livekit.agents")

    class _Agent:
        def __init__(self, instructions=None, llm=None, tts=None, stt=None, chat_ctx=None, **kw):
            self.instructions = instructions
            self._llm = llm
            self._tts = tts
            self.chat_ctx = chat_ctx
            self.session = MagicMock()

    agents.Agent = _Agent
    agents.AgentServer = MagicMock()
    agents.AgentSession = MagicMock()
    agents.ChatContext = MagicMock()
    agents.JobContext = MagicMock()
    agents.JobProcess = MagicMock()
    agents.RunContext = MagicMock()
    agents.cli = MagicMock()
    agents.metrics = MagicMock()

    agents_job = types.ModuleType("livekit.agents.job")
    agents_job.get_job_context = MagicMock()

    agents_llm = types.ModuleType("livekit.agents.llm")

    def function_tool(fn):
        return fn

    agents_llm.function_tool = function_tool

    agents_voice = types.ModuleType("livekit.agents.voice")
    agents_voice.MetricsCollectedEvent = MagicMock()

    plugins = types.ModuleType("livekit.plugins")
    plugins.deepgram = MagicMock()
    plugins.openai = MagicMock()
    plugins.silero = MagicMock()
    plugins.noise_cancellation = MagicMock()

    # Stub out models_ai so agent.py's module-level build_* calls return mocks
    models_ai_mod = types.ModuleType("models_ai")
    models_ai_mod.build_llm = MagicMock(return_value=MagicMock())
    models_ai_mod.build_stt = MagicMock(return_value=MagicMock())
    models_ai_mod.build_tts = MagicMock(return_value=MagicMock())

    return {
        "dotenv": dotenv_mod,
        "models_ai": models_ai_mod,
        "livekit": livekit,
        "livekit.api": livekit_api,
        "livekit.agents": agents,
        "livekit.agents.job": agents_job,
        "livekit.agents.llm": agents_llm,
        "livekit.agents.voice": agents_voice,
        "livekit.plugins": plugins,
        "livekit.plugins.deepgram": MagicMock(),
        "livekit.plugins.openai": MagicMock(),
        "livekit.plugins.silero": MagicMock(),
        "livekit.plugins.noise_cancellation": MagicMock(),
    }


@pytest.fixture(scope="module")
def agent_module():
    """Import agent.py with LiveKit stubs if the real packages aren't present."""
    import models as _models  # noqa: F401

    try:
        import livekit  # noqa: F401
        import agent as m
    except ImportError:
        stubs = _make_livekit_stubs()
        with patch.dict(sys.modules, stubs):
            for mod_name in ("agent",):
                sys.modules.pop(mod_name, None)
            import agent as m

    m.InterviewData = _models.InterviewData
    return m


# ── IntroAgent ──────────────────────────────────────────────────────────────

def test_intro_agent_instantiates(agent_module):
    agent = agent_module.IntroAgent()
    assert agent is not None


def test_intro_agent_has_instructions(agent_module):
    agent = agent_module.IntroAgent()
    assert agent.instructions is not None
    assert len(agent.instructions) > 0


def test_intro_agent_instructions_contain_persona(agent_module):
    import yaml
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
    agent = agent_module.IntroAgent()
    assert cfg["persona"]["name"] in agent.instructions
    assert cfg["persona"]["company"] in agent.instructions


def test_intro_agent_has_name_captured_tool(agent_module):
    assert hasattr(agent_module.IntroAgent, "name_captured")
    assert callable(agent_module.IntroAgent.name_captured)


def test_intro_agent_has_intro_captured_tool(agent_module):
    assert hasattr(agent_module.IntroAgent, "intro_captured")
    assert callable(agent_module.IntroAgent.intro_captured)


def test_intro_agent_no_longer_has_information_gathered(agent_module):
    """Old single-tool capture was replaced by two-step capture."""
    assert not hasattr(agent_module.IntroAgent, "information_gathered")


# ── ExperienceAgent ─────────────────────────────────────────────────────────

def test_experience_agent_instantiates(agent_module):
    agent = agent_module.ExperienceAgent("Alice")
    assert agent is not None


def test_experience_agent_instructions_contain_name(agent_module):
    agent = agent_module.ExperienceAgent("Alice")
    assert "Alice" in agent.instructions


def test_experience_agent_includes_intro_when_provided(agent_module):
    agent = agent_module.ExperienceAgent("Bob", intro="CS grad, 2 years at Acme")
    assert "CS grad, 2 years at Acme" in agent.instructions


def test_experience_agent_lists_topic_openers(agent_module):
    """Agent prompt must enumerate the configured topic openers."""
    agent = agent_module.ExperienceAgent("Alice")
    for opener in agent_module.EXPERIENCE_TOPIC_OPENERS.values():
        assert opener in agent.instructions


def test_experience_agent_accepts_chat_ctx(agent_module):
    mock_ctx = MagicMock()
    agent = agent_module.ExperienceAgent("Bob", chat_ctx=mock_ctx)
    assert agent is not None


def test_experience_agent_has_experience_complete_tool(agent_module):
    assert hasattr(agent_module.ExperienceAgent, "experience_complete")
    assert callable(agent_module.ExperienceAgent.experience_complete)


def test_experience_agent_no_longer_ends_interview(agent_module):
    """Closing happens in TechnicalAgent now — old interview_finished is gone."""
    assert not hasattr(agent_module.ExperienceAgent, "interview_finished")


# ── TechnicalAgent ──────────────────────────────────────────────────────────

def test_technical_agent_instantiates(agent_module):
    agent = agent_module.TechnicalAgent("Alice", "Worked at Acme on ML.")
    assert agent is not None


def test_technical_agent_instructions_contain_summary(agent_module):
    agent = agent_module.TechnicalAgent("Alice", "Worked at Acme on ML.")
    assert "Worked at Acme on ML." in agent.instructions


def test_technical_agent_instructions_reference_role(agent_module):
    import yaml
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
    agent = agent_module.TechnicalAgent("Alice", "summary")
    assert cfg["persona"]["interviewing_for"] in agent.instructions


def test_technical_agent_accepts_chat_ctx(agent_module):
    mock_ctx = MagicMock()
    agent = agent_module.TechnicalAgent("Alice", "summary", chat_ctx=mock_ctx)
    assert agent is not None


def test_technical_agent_has_record_observation_tool(agent_module):
    assert hasattr(agent_module.TechnicalAgent, "record_observation")
    assert callable(agent_module.TechnicalAgent.record_observation)


def test_technical_agent_has_interview_complete_tool(agent_module):
    assert hasattr(agent_module.TechnicalAgent, "interview_complete")
    assert callable(agent_module.TechnicalAgent.interview_complete)


# ── name_captured / intro_captured handoff ──────────────────────────────────

@pytest.mark.asyncio
async def test_name_captured_stores_name_only(agent_module):
    """name_captured stores name but does NOT advance the stage."""
    intro = agent_module.IntroAgent()
    context = MagicMock()
    context.userdata = agent_module.InterviewData()

    result = await intro.name_captured(context, name="Carol")

    assert context.userdata.name == "Carol"
    assert context.userdata.exp is None
    # name_captured does not return a new agent — stage stays put
    assert result is None


@pytest.mark.asyncio
async def test_intro_captured_stores_exp_and_returns_experience_agent(agent_module):
    intro = agent_module.IntroAgent()
    context = MagicMock()
    context.userdata = agent_module.InterviewData(name="Carol")

    result = await intro.intro_captured(context, exp="CS grad")

    assert context.userdata.exp == "CS grad"
    assert isinstance(result, agent_module.ExperienceAgent)


@pytest.mark.asyncio
async def test_intro_captured_passes_chat_ctx(agent_module):
    """chat_ctx propagates from IntroAgent → ExperienceAgent."""
    from unittest.mock import PropertyMock

    intro = agent_module.IntroAgent()
    context = MagicMock()
    context.userdata = agent_module.InterviewData(name="Dan")

    # chat_ctx is a read-only property on the real Agent class — patch it
    fake_ctx = MagicMock()
    with patch.object(
        type(intro), "chat_ctx", new_callable=PropertyMock, return_value=fake_ctx
    ):
        result = await intro.intro_captured(context, exp="5 years at Acme")

    assert isinstance(result, agent_module.ExperienceAgent)


# ── experience_complete handoff ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_experience_complete_stores_summary_and_returns_technical_agent(agent_module):
    exp_agent = agent_module.ExperienceAgent("Alice")
    context = MagicMock()
    context.userdata = agent_module.InterviewData(name="Alice", exp="CS grad")

    result = await exp_agent.experience_complete(
        context, summary="Worked at Acme on ML pipelines for 2 years."
    )

    assert context.userdata.experience_summary == "Worked at Acme on ML pipelines for 2 years."
    assert isinstance(result, agent_module.TechnicalAgent)


# ── record_observation ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_observation_appends_to_notes(agent_module):
    tech = agent_module.TechnicalAgent("Alice", "summary")
    context = MagicMock()
    context.userdata = agent_module.InterviewData()

    await tech.record_observation(
        context,
        question="Explain attention.",
        answer="It's a way to weight inputs.",
        observation="Mentioned weighting but not Q/K/V.",
    )

    assert len(context.userdata.technical_notes) == 1
    note = context.userdata.technical_notes[0]
    assert note["q"] == "Explain attention."
    assert note["a"] == "It's a way to weight inputs."
    assert note["obs"] == "Mentioned weighting but not Q/K/V."


@pytest.mark.asyncio
async def test_record_observation_accumulates(agent_module):
    tech = agent_module.TechnicalAgent("Alice", "summary")
    context = MagicMock()
    context.userdata = agent_module.InterviewData()

    await tech.record_observation(context, question="Q1", answer="A1", observation="O1")
    await tech.record_observation(context, question="Q2", answer="A2", observation="O2")

    assert len(context.userdata.technical_notes) == 2


# ── interview_complete ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_interview_complete_calls_room_delete(agent_module):
    from unittest.mock import PropertyMock

    tech = agent_module.TechnicalAgent("Alice", "summary")
    context = MagicMock()
    context.userdata = agent_module.InterviewData(name="Alice")

    mock_session = MagicMock()
    mock_session.interrupt = MagicMock()
    mock_session.generate_reply = AsyncMock()

    mock_job_ctx = MagicMock()
    mock_job_ctx.api.room.delete_room = AsyncMock()
    mock_job_ctx.room.name = "test-room"

    # `session` is a read-only property on the real Agent class — patch it
    with patch.object(
        type(tech), "session", new_callable=PropertyMock, return_value=mock_session
    ), patch.object(agent_module, "get_job_context", return_value=mock_job_ctx):
        await tech.interview_complete(context)

    mock_session.interrupt.assert_called_once()
    mock_session.generate_reply.assert_called_once()
    mock_job_ctx.api.room.delete_room.assert_called_once()


# ── prompt regression: [Hard rules] block ───────────────────────────────────

def test_intro_agent_has_hard_rules_block(agent_module):
    agent = agent_module.IntroAgent()
    assert "[Hard rules]" in agent.instructions
    assert "Never compliment" in agent.instructions
    assert "Never reuse a question" in agent.instructions


def test_experience_agent_has_hard_rules_block(agent_module):
    agent = agent_module.ExperienceAgent("Alice")
    assert "[Hard rules]" in agent.instructions
    assert "Never compliment" in agent.instructions


def test_technical_agent_has_hard_rules_block(agent_module):
    agent = agent_module.TechnicalAgent("Alice", "summary")
    assert "[Hard rules]" in agent.instructions
    assert "Never compliment" in agent.instructions


def test_experience_agent_has_silence_block(agent_module):
    """Stages where the candidate does most of the talking get a silence block."""
    agent = agent_module.ExperienceAgent("Alice")
    assert "[Silence handling]" in agent.instructions


def test_technical_agent_has_silence_block(agent_module):
    agent = agent_module.TechnicalAgent("Alice", "summary")
    assert "[Silence handling]" in agent.instructions


# ── common_instructions ──────────────────────────────────────────────────────

def test_common_instructions_not_empty(agent_module):
    assert agent_module.common_instructions
    assert len(agent_module.common_instructions) > 50


def test_common_instructions_contains_persona(agent_module):
    import yaml
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
    instr = agent_module.common_instructions
    assert cfg["persona"]["name"] in instr
    assert cfg["persona"]["company"] in instr
    assert cfg["persona"]["interviewing_for"] in instr


# ── build_instructions helper ───────────────────────────────────────────────

def test_build_instructions_includes_role_and_goal(agent_module):
    out = agent_module.build_instructions("Test goal here.")
    assert "[Role]" in out
    assert "[Goal] Test goal here." in out
    assert "[Hard rules]" in out
    assert "[Turn discipline]" in out
    assert "Never speak or print tool names" in out


def test_build_instructions_silence_block_optional(agent_module):
    without = agent_module.build_instructions("Goal", include_silence_block=False)
    with_block = agent_module.build_instructions("Goal", include_silence_block=True)
    assert "[Silence handling]" not in without
    assert "[Silence handling]" in with_block


def test_extract_pseudo_function_call_parses_markup(agent_module):
    text = (
        'Got it. <function=experience_complete>{"summary":"Built a RAG chatbot."}'
        "</function>"
    )
    parsed = agent_module._extract_pseudo_function_call(text)
    assert parsed is not None
    fn_name, args, cleaned = parsed
    assert fn_name == "experience_complete"
    assert args["summary"] == "Built a RAG chatbot."
    assert cleaned == "Got it."


@pytest.mark.asyncio
async def test_apply_pseudo_function_fallback_updates_agent(agent_module):
    session = MagicMock()
    session.userdata = agent_module.InterviewData(name="Dakshi")
    session.current_agent = MagicMock()
    session.current_agent.chat_ctx = MagicMock()

    await agent_module._apply_pseudo_function_fallback(
        session,
        "experience_complete",
        {"summary": "Built ML features for a chatbot."},
    )

    assert session.userdata.experience_summary == "Built ML features for a chatbot."
    session.update_agent.assert_called_once()
    next_agent = session.update_agent.call_args.args[0]
    assert isinstance(next_agent, agent_module.TechnicalAgent)


@pytest.mark.asyncio
async def test_apply_pseudo_function_fallback_records_observation(agent_module):
    session = MagicMock()
    session.userdata = agent_module.InterviewData()
    session.current_agent = MagicMock()

    await agent_module._apply_pseudo_function_fallback(
        session,
        "record_observation",
        {
            "question": "Explain RAG.",
            "answer": "Retrieval plus generation.",
            "observation": "Defined the high-level idea succinctly.",
        },
    )

    assert session.userdata.technical_notes == [
        {
            "q": "Explain RAG.",
            "a": "Retrieval plus generation.",
            "obs": "Defined the high-level idea succinctly.",
        }
    ]
