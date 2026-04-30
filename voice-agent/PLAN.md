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
- Configured in `config.yaml` with `provider: elevenlabs`, an ElevenLabs `model`, and `voice_id`

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

## Phase 2.5 — Agent Logic & Prompting Improvements ⬜ PROPOSED

**Goal:** Make the existing 2-agent flow feel like a real interview — focused single-question turns, active acknowledgement, adaptive follow-ups — and add a third **TechnicalAgent** for role-specific evaluation. This is the last LiveKit-native iteration before Phase 3 replaces orchestration with LangGraph.

### Why now

The current prompts have several issues observable in dev sessions:
1. **`IntroAgent`** asks for *name and introduction* in one breath — candidates often answer one but not the other, and `information_gathered` then either fires with `exp=""` or fires too late.
2. **`ExperienceAgent`** is told "do not ask follow-up questions" — this makes the interview feel mechanical and surfaces no signal beyond a single monologue.
3. The interview ends after one stage; there is no role-specific evaluation, so the persona's "interviewing for *junior AI Developer*" framing is wasted.
4. Silence handling is encoded into the LLM instruction — fragile, not deterministic. (Phase 7 will fix this properly via LangGraph; until then, we tighten the prompt language.)

### 2.5a — Refactor `common_instructions` and persona block

Move from a flat string to a structured prompt template with named sections:

```
[Role]   You are {persona.name}, a {persona.title} at {persona.company}.
[Goal]   Conduct a {stage_name} interview for the {persona.interviewing_for} role.
[Style]  Polite and formal. One question at a time. Acknowledge before asking next.
[Hard rules]
  - Never answer the candidate's questions about the role, the company, or yourself.
  - Never reveal a model answer or hint at one.
  - Never reuse a question already asked in this conversation.
  - Never compliment the candidate's answer ("great", "perfect"); stay neutral.
[Turn discipline]
  - Speak only after the candidate has finished their previous turn.
  - If the candidate's response is empty or off-topic, briefly redirect.
```

The hard-rules block is the highest-leverage fix — current prompt has these scattered or missing.

### 2.5b — `IntroAgent` rewrite

**Behavior change:** ask name first, *wait*, then ask for intro. Two function tools instead of one:

| Tool | Trigger | Stores |
|---|---|---|
| `name_captured(name)` | Candidate states their name | `userdata.name` |
| `intro_captured(exp)`  | Candidate gives a 2+ sentence self-intro | `userdata.exp`, hands off to `ExperienceAgent` |

Splitting captures means `userdata.name` is reliably set before the experience stage starts, so `ExperienceAgent` can address the candidate by name from turn one.

Add an explicit re-prompt branch: if the candidate gives only a name (one word) when an intro was asked, re-ask: "Thanks {name}. Could you tell me a bit about your background — school, what you've worked on?"

### 2.5c — `ExperienceAgent` rewrite

**Behavior change:** ask focused questions, allow up to **2 follow-ups per topic**, then move on.

Topics to cover (pulled from `config.yaml` so the persona can be reused):

```yaml
interview:
  experience_topics:
    - most_recent_role         # "Walk me through your most recent role."
    - one_project_deep_dive    # "Pick one project you're proud of — what was your contribution?"
    - challenge_and_resolution # "Describe a technical challenge you hit and how you resolved it."
```

For each topic, the agent runs a small loop:
1. Ask the topic's opener question.
2. Listen.
3. If the answer is < 20 words OR vague (no specifics), ask one clarifying follow-up — *one*, not a chain.
4. Acknowledge briefly ("Got it." / "Understood.") and move to the next topic.

After all topics are covered, hand off to `TechnicalAgent` via a new function tool `experience_complete(summary)` instead of ending the interview. The `summary` is a one-paragraph LLM-generated summary of what the candidate said — passed to `TechnicalAgent` so its first question can reference real prior context.

### 2.5d — New `TechnicalAgent`

**Purpose:** ask 2 role-specific technical questions, adapted to the candidate's stated experience.

**Inputs:** `name`, `experience_summary`, and the persona's `interviewing_for` role.

**Question generation strategy:**
- Question 1: a *concept* question (e.g., for an AI Dev role: "Walk me through what happens when a transformer attends to a sequence — at a high level.")
- Question 2: an *applied / scenario* question (e.g., "Suppose your fine-tuned model is hallucinating on out-of-distribution inputs in production. Walk me through how you'd debug it.")

Question text is generated by the LLM at runtime, conditioned on:
- Role from `config.yaml`
- `experience_summary` from `ExperienceAgent`

This keeps the agent honest — questions are not hardcoded, but the *kind* of question is constrained by prompt template, so the LLM can't drift into trivia or arithmetic.

**One follow-up rule:** if the candidate's first answer is shallow, ask exactly one clarifying probe. Then move on. Never argue with the answer.

**Internal notes:** after each candidate answer, the agent calls a tool `record_observation(question, answer, observation)` that appends to `userdata.technical_notes: list[dict]`. These notes lay the groundwork for Phase 5 scoring — for now they're just logged.

After both questions, hand off to closing via `interview_complete()` which:
1. Generates a neutral close ("Thank you, {name}. We'll be in touch.")
2. Deletes the room.

### 2.5e — Update `models.py`

```python
@dataclass
class InterviewData:
    name: str | None = None
    exp: str | None = None                          # raw self-intro from IntroAgent
    experience_summary: str | None = None           # one-paragraph summary from ExperienceAgent
    technical_notes: list[dict] = field(default_factory=list)
    # technical_notes entries: {"q": str, "a": str, "obs": str}
    # prev_org / prev_role removed — never populated, dead fields
```

### 2.5f — Update `config.yaml`

```yaml
interview:
  silence_timeout_seconds: 20
  experience_topics:
    - most_recent_role
    - one_project_deep_dive
    - challenge_and_resolution
  technical:
    num_questions: 2
    follow_ups_allowed: 1
```

### 2.5g — Tests

Extend `tests/test_agents.py`:
- `IntroAgent`: `name_captured` stores name without ending stage; `intro_captured` ends stage and produces `ExperienceAgent`.
- `ExperienceAgent`: `experience_complete(summary)` populates `userdata.experience_summary` and returns a `TechnicalAgent`.
- `TechnicalAgent`: `record_observation` appends to `userdata.technical_notes`; `interview_complete` triggers room deletion path (mocked).
- Prompt-level: each agent's `instructions` contains the [Hard rules] block (regression guard so future edits don't drop it).

Target: ~12 new tests, total ~40.

### 2.5h — Manual validation checklist

After implementation, run `uv run agent.py console` and verify:
- [ ] Agent asks name first, waits, then asks for intro (does not combine)
- [ ] Agent never compliments answers ("great", "excellent")
- [ ] Each experience topic gets at most one follow-up
- [ ] Handoff to `TechnicalAgent` happens after the third experience topic
- [ ] Technical questions reference something the candidate actually said
- [ ] Closing message is neutral (no "you did great")

### Deliverable

A demo session in console mode that *feels* like a real interview: paced, focused, role-aware, and 3 stages deep. No LangGraph yet (that's Phase 3), but the prompting bones are solid and the third agent is in place ready to be migrated.

### Risks / non-goals

- **Not adding scoring yet** — `technical_notes` is just structured logs. Real scoring is Phase 5.
- **Silence handling stays prompt-based** for now — Phase 7 makes it deterministic.
- **No resume input yet** — questions are generated from in-conversation context only. Phase 4 adds resume grounding.

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
| 2.5 | ⬜ Next | Agent logic + prompting overhaul; add `TechnicalAgent` (3-stage flow) |
| 3 | ⬜ | LangGraph orchestration layer + SQLite checkpointing |
| 4 | ⬜ | Resume ingestion (PDF → personalized questions) |
| 5 | ⬜ | Per-answer scoring + post-interview report |
| 6 | ⬜ | Next.js frontend |
| 7 | ⬜ | Deterministic silence handling via LangGraph |
