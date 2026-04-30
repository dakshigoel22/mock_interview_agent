# Mock Interview Agent

A voice-based mock interview system built on LiveKit. The current implementation lives in [`voice-agent/`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent) and runs a staged AI interview over voice using Groq for LLM + STT and ElevenLabs for TTS.

## Current Flow

The voice pipeline is:

```text
Candidate (voice)
   -> LiveKit Room
   -> LiveKit Agent
   -> Graph-based interview orchestrator
   -> Agent speech back to candidate
```

The interview is currently structured as:

1. name capture
2. background / self-introduction
3. experience questions
4. technical questions
5. report generation
6. room close

The control flow is now handled through the graph orchestration layer in [`voice-agent/graph/`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/graph), while LiveKit remains responsible for voice transport, STT/TTS, and room lifecycle.

## Stack

| Layer | Provider / Library | Notes |
|---|---|---|
| Voice transport | LiveKit | real-time room + agent runtime |
| Orchestration | Graph-based pipeline | implemented in `voice-agent/graph` |
| LLM | Groq | `llama-3.3-70b-versatile` |
| STT | Groq Whisper | `whisper-large-v3-turbo` |
| TTS | ElevenLabs | configurable model + `voice_id` |
| VAD | Silero | prewarmed in the LiveKit worker |

## Project Layout

```text
mock_interview_agent/
├── README.md
└── voice-agent/
    ├── agent.py
    ├── config.yaml
    ├── models.py
    ├── graph/
    │   ├── __init__.py
    │   ├── interview_graph.py
    │   ├── nodes.py
    │   └── state.py
    ├── models_ai/
    │   ├── __init__.py
    │   ├── audio.py
    │   ├── stt.py
    │   └── tts.py
    ├── tests/
    ├── PLAN.md
    ├── PROJECT_STATUS.md
    ├── pyproject.toml
    ├── requirements.txt
    └── .env.example
```

## Interview State

Runtime user/session data is stored in [`voice-agent/models.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models.py):

```python
@dataclass
class InterviewData:
    name: str | None = None
    exp: str | None = None
    experience_summary: str | None = None
    technical_notes: list[dict] = field(default_factory=list)
```

The graph orchestration layer also maintains per-session structured state and writes a saved artifact to:

- [`voice-agent/session_logs/`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/session_logs)

Each saved session JSON currently includes the captured candidate details, transcript, experience summary, technical observations, lightweight scores, and generated report markdown.

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- a LiveKit project or self-hosted LiveKit server
- a [Groq API key](https://console.groq.com)
- an [ElevenLabs API key](https://elevenlabs.io)

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

Fill in `.env.local` with your real secrets:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

Edit [`voice-agent/config.yaml`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/config.yaml) to customize:

- interviewer persona
- role being interviewed for
- Groq model names
- ElevenLabs model + `voice_id`
- experience topic sequence
- number of technical questions

## Run

From inside `voice-agent/`:

```bash
uv run agent.py dev
```

Useful alternatives:

```bash
uv run agent.py console
uv run pytest tests/ -v
```

## Deploy

If you use the LiveKit CLI from repo root, point it at the nested secrets file:

```bash
lk agent deploy --secrets-file voice-agent/.env.local
```

If you are already inside `voice-agent/`, use:

```bash
lk agent deploy --secrets-file .env.local
```

## Current Capabilities

- Captures candidate name and follows up for background
- Uses candidate details in later prompts
- Walks through configured experience topics
- Asks technical questions after the experience stage
- Records technical observations
- Generates a simple end-of-session report artifact
- Saves per-session logs under `session_logs/`
- Closes the room when the interview reaches completion

## Current Limitations

- The graph orchestration is implemented, but the broader persistence/reporting stack is still local-file based rather than a full database-backed product workflow
- Scoring is currently lightweight and heuristic, not yet a dedicated judge-model evaluation pipeline
- Resume ingestion and a frontend report viewer are not implemented yet

## Docs

- Project status: [`voice-agent/PROJECT_STATUS.md`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/PROJECT_STATUS.md)
- Roadmap: [`voice-agent/PLAN.md`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/PLAN.md)

## Tests

Run from `voice-agent/`:

```bash
uv run pytest tests/ -v
```

The repo currently includes coverage for:

- config validation
- agent prompt / handoff behavior
- graph-node progression
- dataclass behavior
- STT/TTS wrapper behavior
- audio conversion helpers
