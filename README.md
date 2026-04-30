# Mock Interview Agent

A voice-based mock interview system built on LiveKit. A candidate connects via browser and completes a structured voice interview conducted by an AI interviewer.

## How It Works

```
Candidate (voice)
       │
  LiveKit Room  (WebRTC)
       │
  LiveKit Agent (Python)
       │
  ┌────┴──────────────────────────┐
  │  IntroAgent                   │  → greets candidate, collects name + intro
  │       ↓ (handoff)             │
  │  ExperienceAgent              │  → asks about past roles, ends interview
  └───────────────────────────────┘
       │
  STT: Groq Whisper  │  LLM: Groq Llama 3.3  │  TTS: ElevenLabs
```

State (`InterviewData`) is shared across agent handoffs via `RunContext`.

## Stack

| Layer | Provider | Model |
|---|---|---|
| LLM | Groq API | Llama 3.3-70b-versatile |
| STT | Groq Whisper | whisper-large-v3-turbo |
| TTS | ElevenLabs | default voice |
| VAD | Silero | — |
| Voice transport | LiveKit | open source |

## Agents

### IntroAgent
- Greets the candidate using the persona defined in `config.yaml`
- Asks for name and a brief self-introduction
- Calls `information_gathered(name, exp)` → hands off to `ExperienceAgent`

### ExperienceAgent
- Asks the candidate to describe past work experiences and internships
- Does not ask follow-up questions
- Handles silence: after 20 s of no response, checks in ("are you still there?"); on second timeout, ends the interview
- Calls `interview_finished(reason)` → generates goodbye, deletes the LiveKit room

## Shared State

```python
@dataclass
class InterviewData:
    name: str | None = None     # candidate's name
    prev_org: str | None = None
    prev_role: str | None = None
    exp: str | None = None      # self-introduction text
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- LiveKit server (self-hosted or [LiveKit Cloud](https://livekit.io/cloud))
- [Groq API key](https://console.groq.com) (free tier — powers LLM + STT)
- [ElevenLabs API key](https://elevenlabs.io) (free tier — powers TTS)

### Install

```bash
git clone https://github.com/dakshigoel22/mock_interview_agent.git
cd mock_interview_agent/voice-agent
uv sync
```

### Configure

```bash
cp .env.example .env.local
```

Fill in `.env.local`:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

Edit `config.yaml` to change the interviewer persona, models, or silence timeout.

### Run

```bash
# Dev mode — connects to LiveKit Cloud, auto-reloads on file changes
uv run agent.py dev

# Console mode — runs a text-only session in the terminal (no LiveKit needed)
uv run agent.py console
```

To self-host LiveKit instead of LiveKit Cloud:
```bash
docker run --rm -p 7880:7880 livekit/livekit-server --dev
# set LIVEKIT_URL=ws://localhost:7880 in .env.local
```

### Test

```bash
uv run pytest tests/ -v
```

## Project Structure

```
voice-agent/
├── agent.py            # IntroAgent + ExperienceAgent, entrypoint
├── models.py           # InterviewData dataclass
├── models_ai/
│   ├── __init__.py     # build_llm / build_stt / build_tts factory functions
│   ├── stt.py          # GroqWhisperSTT — custom LiveKit STT plugin
│   ├── tts.py          # HFKokoroTTS — Kokoro wrapper (available, not active)
│   └── audio.py        # PCM ↔ float32 conversion helpers
├── config.yaml         # Persona, model settings, silence timeout
├── tests/
│   ├── test_config.py
│   ├── test_dataclass.py
│   ├── test_agents.py
│   └── test_models_ai.py
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Cleanup, config extraction, 28 tests |
| 2 | ✅ Done | Model swap — Groq LLM + STT, ElevenLabs TTS |
| 3 | ⬜ Next | LangGraph orchestration + SQLite checkpointing |
| 4 | ⬜ | Resume ingestion (PDF → personalized questions) |
| 5 | ⬜ | Per-answer scoring + post-interview report |
| 6 | ⬜ | Next.js frontend |
| 7 | ⬜ | Deterministic silence handling via LangGraph |

See [`voice-agent/PLAN.md`](voice-agent/PLAN.md) for the full implementation plan.
