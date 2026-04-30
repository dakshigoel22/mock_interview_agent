# Mock Interview Agent — Full Implementation Plan

## Project Goal

Transform the current 2-agent voice prototype into a fully self-hostable, open source mock interview platform. A candidate uploads a resume, completes a voice interview (behavioral + technical rounds), and receives a scored report — all without any paid API dependency.

## Architecture Overview

```
Candidate
   │ (voice)
   ▼
LiveKit Room  (open source WebRTC)
   │ (STT: faster-whisper)
   ▼
LiveKit Agent (Python)
   │ candidate_utterance
   ▼
┌─────────────────────────────────────────┐
│         LangGraph Interview Graph       │
│                                         │
│  intro → behavioral → technical         │
│               ↘ evaluation → report    │
└─────────────────────────────────────────┘
   │ next_question / action
   ▼
LiveKit Agent
   │ (TTS: Kokoro)
   ▼
Candidate
```

**Key principle:** LangGraph owns interview logic and state. LiveKit owns real-time voice transport. They are complementary — the LiveKit agent calls `graph.astream()` on each candidate turn and speaks whatever the graph emits.

LangGraph state is checkpointed to SQLite — persistent across restarts, inspectable, replayable.

---

## Open Source Stack

| Layer | Phase 1 (original) | Current (Phase 2) | Phase 3+ target |
|---|---|---|---|
| LLM | OpenAI GPT-4.1-mini | **Groq — Llama 3.3-70b-versatile** | same |
| STT | Deepgram Nova-3 | **Groq Whisper large-v3-turbo** | same |
| TTS | Cartesia Sonic-3 | **ElevenLabs** (free tier) | Kokoro-82M (HF, Apache 2) |
| Orchestration | LiveKit function_tool handoffs | same | **LangGraph** |
| Persistence | None (in-memory) | None | **SQLite** + LangGraph checkpointer |
| Frontend | None | None | **Next.js** + LiveKit React components |
| Resume parsing | None | None | **PyMuPDF** |
| Voice infra | LiveKit | LiveKit | LiveKit |

---

## Target Folder Structure

```
mock_interview_agent/
├── voice-agent/
│   ├── agent.py                  # LiveKit agent entrypoint (thin)
│   ├── models.py                 # InterviewData + shared types
│   ├── config.yaml               # Persona, models, interview settings
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py              # InterviewState TypedDict
│   │   ├── nodes.py              # All LangGraph nodes
│   │   └── interview_graph.py    # Graph assembly + SQLite checkpointer
│   ├── models_ai/
│   │   ├── stt.py                # faster-whisper LiveKit plugin wrapper
│   │   └── tts.py                # Kokoro LiveKit plugin wrapper
│   ├── api/
│   │   ├── main.py               # FastAPI: resume upload, report retrieval
│   │   └── db.py                 # SQLite session storage
│   ├── tests/
│   │   ├── test_config.py
│   │   ├── test_dataclass.py
│   │   ├── test_agents.py
│   │   ├── test_graph_nodes.py   # (added in Phase 3)
│   │   └── test_api.py           # (added in Phase 5)
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── PLAN.md
└── frontend/
    ├── app/
    │   ├── page.tsx              # Landing / resume upload
    │   ├── interview/[room]/     # Live voice room + transcript
    │   └── report/[id]/          # Post-interview scorecard
    └── package.json
```

---

## Phase 1 — Cleanup ✅ DONE

**Goal:** Remove dead code, curate dependencies, extract all hardcoded config.

- [x] Delete `agent_withfallback.py` (500 lines of dead, incompatible code)
- [x] Rewrite `requirements.txt` (200 unrelated packages → 14 curated)
- [x] Update `pyproject.toml` (fix project name, add pyyaml + pytest extras)
- [x] Create `config.yaml` (persona name/company/role, model strings, silence timeout)
- [x] Extract `InterviewData` dataclass to `models.py` (no LiveKit dependency)
- [x] Rewrite `agent.py` (remove ~150 lines of commented-out dead code, fix `load_dotenv`, load config, rename `Prev_experience_Agent` → `ExperienceAgent`)
- [x] Create `.env.example` (credential template)
- [x] Create `.gitignore` (protect `.env.local`, `venv/`)
- [x] Write and pass 28 tests (`test_config`, `test_dataclass`, `test_agents`)

---

## Phase 2 — Model Swap ✅ DONE

**Goal:** Replace the original paid APIs (OpenAI, Deepgram, Cartesia) with free-tier / open-source equivalents.

### 2a — STT: Groq Whisper ✅

Wrote a custom LiveKit-compatible STT plugin at `models_ai/stt.py` (`GroqWhisperSTT`).
- Sends a WAV clip to Groq's Whisper API (`/openai/v1/audio/transcriptions`)
- Wrapped by LiveKit's `StreamAdapter` + Silero VAD for streaming support
- Model: `whisper-large-v3-turbo` (configurable via `config.yaml`)
- Requires: `GROQ_API_KEY`

### 2b — TTS: ElevenLabs ✅

Uses `livekit-plugins-elevenlabs` (`build_tts` in `models_ai/__init__.py`).
- Free tier available at elevenlabs.io
- Requires: `ELEVENLABS_API_KEY`

> **Note:** `models_ai/tts.py` contains a `HFKokoroTTS` wrapper (Kokoro-82M via HuggingFace Inference API, Apache 2) that is implemented but not currently active. To switch, replace `build_tts` to return `HFKokoroTTS(hf_token=...)` and set `HF_TOKEN` in `.env.local`.

### 2c — LLM: Groq (Llama 3.3) ✅

Uses `livekit-plugins-openai` pointed at Groq's OpenAI-compatible endpoint.
- Model: `llama-3.3-70b-versatile` (450+ tok/s on free tier)
- Base URL: `https://api.groq.com/openai/v1`
- Requires: `GROQ_API_KEY`

### 2d — Audio helpers ✅

`models_ai/audio.py`: int16 PCM ↔ float32 conversion utilities for Whisper input/output.

### 2e — Tests ✅

`tests/test_models_ai.py`:
- STT wrapper initializes and sets correct model name
- TTS (ElevenLabs) integrates via plugin
- Audio conversion round-trips correctly

**Deliverable:** Agent runs with only `GROQ_API_KEY` + `ELEVENLABS_API_KEY` + LiveKit credentials. No local GPU or heavy model downloads required.

---

## Phase 3 — LangGraph Orchestration

**Goal:** Replace implicit function_tool handoffs with an explicit, inspectable state machine.

### 3a — InterviewState

Define in `graph/state.py`:

```python
class InterviewState(TypedDict):
    session_id: str
    candidate_name: str
    resume_text: str
    stage: Literal["intro", "behavioral", "technical", "evaluation", "done"]
    transcript: list[dict]          # [{"role": "agent"|"candidate", "text": str}]
    scores: list[dict]              # [{"question": str, "answer": str, "score": int, "reason": str}]
    current_question: str
    answer_count: int
    last_response_ts: float
```

### 3b — Graph nodes (`graph/nodes.py`)

| Node | Input | Output | Description |
|---|---|---|---|
| `intro_node` | state | state + current_question | Greets candidate, asks name |
| `behavioral_node` | state | state + current_question | Generates behavioral question from resume context |
| `technical_node` | state | state + current_question | Generates role-specific technical question |
| `silence_check_node` | state | state | Routes to check-in or done based on `last_response_ts` |
| `evaluation_node` | state | state + scores entry | Scores the last answer (1–5) using a judge LLM call |
| `report_node` | state | state (terminal) | Aggregates scores, writes report to SQLite |

### 3c — Graph assembly (`graph/interview_graph.py`)

```
START → intro_node → behavioral_node (×N) → technical_node (×N)
                                                  ↓
                                          evaluation_node
                                                  ↓
                                           report_node → END
```

Conditional edges:
- After `behavioral_node`: if `answer_count < MAX_BEHAVIORAL` → loop back, else → `technical_node`
- After `technical_node`: if `answer_count < MAX_TECHNICAL` → loop back, else → `evaluation_node`
- After any node: if silence timeout exceeded → `silence_check_node` → done or re-prompt

Use `SqliteSaver` checkpointer: `SqliteSaver.from_conn_string("interviews.db")`

### 3d — Wire LangGraph into LiveKit agent

In `agent.py`, the single `InterviewAgent` class calls `graph.astream()` on each user turn:

```python
async def on_user_turn(self, text: str):
    state_update = {"transcript": [...], "last_response_ts": time.time()}
    async for chunk in graph.astream(state_update, config={"configurable": {"thread_id": session_id}}):
        await self.session.say(chunk["current_question"])
```

This replaces the `IntroAgent` / `ExperienceAgent` multi-class handoff pattern.

### 3e — Update tests

Add `tests/test_graph_nodes.py`:
- Test each node with a mock LLM (no real Ollama needed)
- Test conditional edge routing
- Test `SqliteSaver` checkpointing round-trip (use `:memory:` SQLite)
- Test silence timeout routing

**Deliverable:** Interview flow is explicit, observable, and resumable on crash.

---

## Phase 4 — Resume Ingestion

**Goal:** Personalize behavioral and technical questions using the candidate's actual background.

### 4a — PDF parsing

Add `PyMuPDF` (`fitz`) to dependencies.
Write `api/resume.py`:

```python
def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)
```

### 4b — FastAPI upload endpoint (`api/main.py`)

```
POST /session/start
  Body: multipart/form-data (resume PDF, candidate name, job role)
  Response: { session_id, livekit_token, room_name }
```

Flow:
1. Parse PDF → extract text
2. Create LangGraph state with `resume_text` populated
3. Generate LiveKit room token
4. Return token to frontend

### 4c — Inject resume into graph

`behavioral_node` and `technical_node` receive `resume_text` in state.
System prompt: *"Based on this resume: {resume_text}, ask one relevant behavioral question about..."*

### 4d — Update tests

- Test PDF extraction with a sample PDF fixture
- Test `/session/start` endpoint (mock PDF, mock LiveKit token generation)
- Test that nodes use `resume_text` in prompt construction

**Deliverable:** Questions are dynamically tailored to the candidate's actual experience.

---

## Phase 5 — Scoring and Post-Interview Report

**Goal:** Every answer is scored; a structured report is generated and persisted at session end.

### 5a — Evaluation node

`evaluation_node` makes a separate LLM call after each candidate answer:

```
System: You are an interview evaluator. Score the following answer on a scale of 1-5.
        Criteria: relevance to question, clarity, depth of insight.
        Return JSON: {"score": int, "reason": str}
User: Question: {question}
      Answer: {answer}
```

Score is appended to `state["scores"]`.

### 5b — Report node

`report_node` runs at the end of the interview:
- Aggregates all scores into an average
- Formats a Markdown report with per-question breakdown
- Saves to SQLite via `api/db.py`:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    candidate_name TEXT,
    report_md TEXT,
    overall_score REAL,
    created_at TIMESTAMP
);
```

### 5c — Report retrieval endpoint

```
GET /report/{session_id}
  Response: { candidate_name, overall_score, report_md, created_at }
```

### 5d — Update tests

- Test `evaluation_node` parses LLM JSON response correctly
- Test `report_node` generates valid Markdown
- Test SQLite write/read round-trip
- Test `GET /report/{session_id}` endpoint

**Deliverable:** Every interview produces a durable, scored report accessible via API.

---

## Phase 6 — Frontend

**Goal:** A browser-accessible UI so anyone can start an interview, hear the agent, and see their report. No technical setup required for the end user.

### Pages

**`/` — Landing page**
- Upload resume (PDF)
- Enter name and target role
- Submit → calls `POST /session/start` → receives LiveKit token
- Redirects to `/interview/[room]`

**`/interview/[room]` — Live interview room**
- Uses `@livekit/components-react` (`LiveKitRoom`, `useVoiceAssistant`)
- Shows live transcript in a sidebar (updated via LiveKit data messages)
- Minimal UI: animated waveform, candidate name, current stage indicator
- On session end → redirects to `/report/[session_id]`

**`/report/[session_id]` — Post-interview report**
- Renders the Markdown report from `GET /report/{session_id}`
- Shows overall score, per-question breakdown with reasons
- Download as PDF button (use `react-pdf` or `window.print()`)

### Tech

- Next.js 14 (App Router)
- `@livekit/components-react` for voice room
- Tailwind CSS for styling
- No additional paid services

### Update tests

- Component smoke tests for Landing and Report pages (Jest + React Testing Library)
- E2E happy path with Playwright: upload PDF → connect to room mock → view report

**Deliverable:** Shareable demo URL. Non-technical stakeholders can use and evaluate the product.

---

## Phase 7 — Silence Handling (Robust)

**Goal:** Replace the current LLM-instruction-based silence hack with deterministic timeout logic inside LangGraph.

### How it works

`last_response_ts` is stored in `InterviewState` and updated on every candidate utterance.

A `silence_check_node` is injected as a conditional edge after every main node:

```python
def route_after_node(state: InterviewState) -> str:
    elapsed = time.time() - state["last_response_ts"]
    if elapsed > config["interview"]["silence_timeout_seconds"]:
        return "silence_check_node"
    return "next_node"
```

`silence_check_node` logic:
- First timeout: generate *"[Name], are you still there?"* and reset `last_response_ts`
- Second timeout (no response after check-in): route to `report_node` and close room

This is explicit, auditable, and testable — no reliance on the LLM following a timing instruction.

### Update tests

- Test `silence_check_node` routes correctly when `last_response_ts` is stale
- Test double-timeout → `report_node` routing
- Test normal flow (no silence) is unaffected

---

## Running the Full Stack (target)

```bash
# 1. Start Ollama with Llama 3.1
ollama pull llama3.1
ollama serve

# 2. Start LiveKit server (Docker)
docker run --rm -p 7880:7880 livekit/livekit-server --dev

# 3. Start the voice agent
cd voice-agent
uv run python agent.py

# 4. Start the API server
uv run uvicorn api.main:app --reload

# 5. Start the frontend
cd ../frontend
npm install && npm run dev
# → open http://localhost:3000
```

---

## Phase Status

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Cleanup, config extraction, 28 tests |
| 2 | ✅ Done | Model swap — Groq Whisper STT, ElevenLabs TTS, Groq Llama 3.3 LLM |
| 3 | ⬜ Next | LangGraph orchestration layer + SQLite checkpointing |
| 4 | ⬜ | Resume ingestion (PDF → personalized questions) |
| 5 | ⬜ | Per-answer scoring + post-interview report |
| 6 | ⬜ | Next.js frontend |
| 7 | ⬜ | Deterministic silence handling via LangGraph |
