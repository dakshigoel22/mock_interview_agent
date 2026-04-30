"""Tests for IntroAgent and ExperienceAgent classes."""
import sys
import types
import pytest
from pathlib import Path
from dataclasses import dataclass
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
    # models.py is pure Python — always importable
    import importlib, models as _models  # noqa: E401

    try:
        import livekit  # noqa: F401
        import agent as m
    except ImportError:
        stubs = _make_livekit_stubs()
        with patch.dict(sys.modules, stubs):
            for mod_name in ("agent",):
                sys.modules.pop(mod_name, None)
            import agent as m

    # Attach InterviewData so tests can reference it via agent_module
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


def test_intro_agent_has_information_gathered_tool(agent_module):
    assert hasattr(agent_module.IntroAgent, "information_gathered")
    assert callable(agent_module.IntroAgent.information_gathered)


# ── ExperienceAgent ──────────────────────────────────────────────────────────

def test_experience_agent_instantiates(agent_module):
    agent = agent_module.ExperienceAgent("Alice")
    assert agent is not None


def test_experience_agent_instructions_contain_name(agent_module):
    agent = agent_module.ExperienceAgent("Alice")
    assert "Alice" in agent.instructions


def test_experience_agent_accepts_chat_ctx(agent_module):
    mock_ctx = MagicMock()
    agent = agent_module.ExperienceAgent("Bob", chat_ctx=mock_ctx)
    assert agent is not None


def test_experience_agent_has_interview_finished_tool(agent_module):
    assert hasattr(agent_module.ExperienceAgent, "interview_finished")
    assert callable(agent_module.ExperienceAgent.interview_finished)


# ── information_gathered handoff ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_information_gathered_stores_data(agent_module):
    intro = agent_module.IntroAgent()
    context = MagicMock()
    context.userdata = agent_module.InterviewData()

    result = await intro.information_gathered(context, name="Carol", exp="CS grad")

    assert context.userdata.name == "Carol"
    assert context.userdata.exp == "CS grad"
    assert isinstance(result, agent_module.ExperienceAgent)


@pytest.mark.asyncio
async def test_information_gathered_passes_chat_ctx(agent_module):
    intro = agent_module.IntroAgent()
    intro.chat_ctx = MagicMock()
    context = MagicMock()
    context.userdata = agent_module.InterviewData()

    result = await intro.information_gathered(context, name="Dan", exp="5 years")

    assert isinstance(result, agent_module.ExperienceAgent)


# ── common_instructions ──────────────────────────────────────────────────────

def test_common_instructions_not_empty(agent_module):
    assert agent_module.common_instructions
    assert len(agent_module.common_instructions) > 50


def test_common_instructions_no_hardcoded_fallback(agent_module):
    import yaml
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
    instr = agent_module.common_instructions
    assert cfg["persona"]["name"] in instr
    assert cfg["persona"]["company"] in instr
    assert cfg["persona"]["interviewing_for"] in instr
