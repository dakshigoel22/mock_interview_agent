# Voice Agent Project Status

Last updated: 2026-04-30

## 1. Project Summary

`voice-agent` is a Python-based mock interview system built on top of LiveKit's agent framework. Its current focus is a real-time voice interview experience where an AI interviewer joins a LiveKit room, asks staged interview questions, listens to spoken answers, and closes the room when the interview is complete.

At the moment, the project is a multi-stage voice interviewer rather than a full hiring platform. It already supports:

- real-time voice interaction through LiveKit
- a staged interview flow with agent handoffs
- configurable interviewer persona and interview settings
- Groq-backed LLM and speech-to-text integrations
- ElevenLabs-backed text-to-speech in the active runtime
- a partially implemented Hugging Face Kokoro TTS wrapper for a more open stack
- automated tests around config, data models, agent prompts, and model wrappers

The broader vision, documented in `PLAN.md`, is to evolve this into a self-hostable mock interview platform with resume upload, technical and behavioral evaluation, persistence, report generation, and eventually a frontend plus LangGraph-based orchestration.

## 2. Current Architecture

The current application is centered around [`agent.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/agent.py), which acts as the LiveKit entrypoint and interview orchestrator.

High-level flow:

1. Environment variables are loaded from `.env` and `.env.local`.
2. Runtime configuration is loaded from [`config.yaml`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/config.yaml).
3. `prewarm()` initializes:
   - Silero VAD
   - LLM via `build_llm()`
   - STT via `build_stt()`
   - TTS via `build_tts()`
4. A LiveKit `AgentSession[InterviewData]` is started.
5. The session begins with `IntroAgent`.
6. Agents hand off stage-to-stage using LiveKit `function_tool`s.
7. At the end of the technical round, the room is deleted through the LiveKit API.

### Core files

- [`agent.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/agent.py): main interview flow, stage definitions, LiveKit startup
- [`models.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models.py): shared interview state dataclass
- [`config.yaml`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/config.yaml): persona, provider, and interview settings
- [`models_ai/__init__.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models_ai/__init__.py): factories for LLM, STT, and TTS
- [`models_ai/stt.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models_ai/stt.py): Groq Whisper wrapper
- [`models_ai/tts.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models_ai/tts.py): Hugging Face Kokoro wrapper
- [`models_ai/audio.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models_ai/audio.py): PCM/float conversion helpers
- [`PLAN.md`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/PLAN.md): long-term roadmap

## 3. Interview Flow

The live interview is implemented as three explicit stages.

### Stage 1: `IntroAgent`

Purpose:
- greet the candidate
- collect the candidate's name first
- collect a short self-introduction second

Important behavior:
- asks only for the name first
- calls `name_captured(name)` after the candidate provides it
- asks for the introduction in a separate turn
- calls `intro_captured(exp)` only after a substantive introduction

This is a stronger design than a single combined capture step because it reduces the chance of missing structured data when the candidate answers one question but not the other.

### Stage 2: `ExperienceAgent`

Purpose:
- walk through configured experience topics
- ask one opener per topic
- optionally ask one clarifying follow-up if the answer is vague or too short
- produce a factual experience summary

The topic order comes from `config.yaml`:

- `most_recent_role`
- `one_project_deep_dive`
- `challenge_and_resolution`

The actual question openers are stored in the `EXPERIENCE_TOPIC_OPENERS` constant so prompt behavior and tests stay aligned.

When the stage finishes, it calls `experience_complete(summary)` and hands off to the technical round.

### Stage 3: `TechnicalAgent`

Purpose:
- ask role-relevant technical questions
- store neutral observations for each answer
- end the interview cleanly

The technical stage is driven by config values:

- `num_questions`
- `follow_ups_allowed`

For each answer, the agent calls `record_observation(question, answer, observation)` and appends data into `InterviewData.technical_notes`. After the configured number of questions are completed, it calls `interview_complete()`, gives a short neutral closing message, and deletes the LiveKit room.

## 4. Prompting Strategy

One of the strongest parts of the current codebase is the shared `build_instructions()` helper in [`agent.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/agent.py). Instead of using one long free-form system prompt, it builds a structured block-based instruction set with sections such as:

- role
- goal
- style
- hard rules
- turn discipline
- optional silence handling

That structure makes the project easier to maintain because:

- persona changes remain centralized in `config.yaml`
- each stage can inject its own goal while preserving consistent interviewer behavior
- tests can validate prompt content more reliably
- future stages can reuse the same prompt contract

The prompt design is explicitly trying to keep the interviewer:

- formal
- neutral
- one-question-at-a-time
- resistant to leaking hints or answers

## 5. Model and Provider Layer

The provider abstraction is intentionally lightweight. Instead of creating a large internal SDK layer, the project uses small builder functions in [`models_ai/__init__.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models_ai/__init__.py).

### Active runtime providers

- LLM: Groq via OpenAI-compatible endpoint
- STT: Groq Whisper
- TTS: ElevenLabs

### Implemented but not active by default

- TTS: Hugging Face Kokoro (`HFKokoroTTS`)

The runtime and config are now aligned on ElevenLabs as the active TTS provider. The Hugging Face Kokoro wrapper remains in the repository as an optional alternative implementation, but it is no longer the primary configured path.

## 6. Shared State Model

The interview state lives in [`models.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models.py) as:

- `name`
- `exp`
- `experience_summary`
- `technical_notes`

This is a clean improvement over flatter early-stage state because it now reflects the actual interview pipeline:

- candidate identity
- intro/background
- condensed behavioral summary
- structured technical observations

Using `field(default_factory=list)` for `technical_notes` is also the right choice and is covered by tests to prevent shared mutable defaults between sessions.

## 7. Testing Coverage

The project has a useful test layer for its current maturity level.

### Covered areas

- config loading and required keys
- `InterviewData` defaults and mutability
- agent prompt content and stage handoff behavior
- model wrapper construction
- audio conversion helpers

### Testing style

The tests use LiveKit stubs heavily so the agent module can be imported and verified without requiring a full LiveKit runtime. That is a pragmatic choice for fast unit tests and makes prompt and handoff logic much easier to iterate on.

### Not yet covered deeply

- end-to-end LiveKit room behavior
- real provider API integration behavior
- resilience around network/provider failures
- transcript quality and audio edge cases
- persistence or reporting, because those layers are not implemented yet

## 8. Implementation Status vs Roadmap

The roadmap in [`PLAN.md`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/PLAN.md) is broader than the code currently present. The project is best understood as being between a cleaned-up prototype and a richer interview engine.

### Already implemented

- cleaned-up project structure
- config-driven persona and provider setup
- staged `IntroAgent -> ExperienceAgent -> TechnicalAgent` flow
- Groq LLM and Groq Whisper integration
- ElevenLabs runtime TTS integration
- Kokoro TTS wrapper implementation
- unit tests for core non-network behavior

### Partially implemented or transitional

- open-source TTS migration: wrapper exists, active factory still uses ElevenLabs
- silence handling: described in prompts, but still dependent on model compliance rather than deterministic control logic
- technical evaluation: observations are stored, but there is no scoring/report pipeline yet

### Planned but not present yet

- LangGraph orchestration
- SQLite persistence/checkpointing
- resume parsing
- report generation
- frontend
- API layer
- full self-hosted interview platform workflow

## 9. Latest Changes Tracking

This section combines the most recent committed milestone with the current uncommitted worktree state.

### Latest commit history

Recent commits seen in git:

- `ce08641` - Phase 1 + 2: cleanup, model swap, tests, docs
- `a85fb69` - Revise README for setup and running instructions
- `b1a6d3b` - Changes

### Current working-tree changes

There are active uncommitted edits in these tracked files:

- [`PLAN.md`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/PLAN.md)
- [`agent.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/agent.py)
- [`config.yaml`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/config.yaml)
- [`models.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/models.py)
- [`tests/test_agents.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/tests/test_agents.py)
- [`tests/test_dataclass.py`](/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/tests/test_dataclass.py)

`git diff --stat` shows:

- 649 insertions
- 65 deletions

### What the current edits are doing

From the inspected diffs and current file contents, the active changes appear to be focused on:

1. expanding the interview from a simpler earlier flow into a three-stage design with `TechnicalAgent`
2. replacing older interview state fields with `experience_summary` and `technical_notes`
3. improving prompt structure with stricter stage-specific instructions
4. adding test coverage for new handoff and state behavior
5. extending the roadmap to describe the newer architecture and next phases in more detail

## 10. Strengths

- Clear staged interview design
- Good separation between configuration, orchestration, and provider wrappers
- Thoughtful prompt structure
- Lightweight, testable state model
- Useful unit tests for a still-evolving codebase
- Ambitious roadmap with a believable path forward

## 11. Main Gaps and Risks

- silence handling still depends on LLM obedience
- no persistence for interview sessions or evaluation outputs
- no scoring or report generation yet despite roadmap goals
- no frontend or public API yet
- duplicate stray files may create maintenance noise

## 12. Recommended Next Steps

If the goal is to stabilize the current version before starting the larger platform work, the highest-value next steps are:

1. decide whether TTS should officially be ElevenLabs or Hugging Face Kokoro right now, then align config, factories, and docs
2. add a concise README or status entry that explains the current three-stage flow and required environment variables
3. add a small integration test path around agent handoff and `InterviewData` accumulation
4. clean duplicate files from the repo tree so the real source of truth is obvious
5. choose whether the next milestone is:
   - productionizing the current LiveKit flow
   - or moving directly into LangGraph + persistence

If the goal is to move toward the long-term product vision, then LangGraph state orchestration and persistent session storage are the next major architectural steps.
