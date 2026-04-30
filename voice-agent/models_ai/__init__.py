"""Factory functions that build STT / TTS / LLM objects from config.yaml."""
from __future__ import annotations

import os
from typing import Any


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        import sys
        if "download-files" in sys.argv:
            return "dummy_key_for_build"
        raise RuntimeError(f"Missing required environment variable: {name}. Check your .env file.")
    return val


def build_stt(cfg: dict[str, Any]):
    """Groq Whisper STT — open source Whisper model served via Groq API."""
    from .stt import GroqWhisperSTT

    return GroqWhisperSTT(
        model=cfg.get("model", "whisper-large-v3-turbo"),
        api_key=_require_env("GROQ_API_KEY"),
    )


def build_tts(cfg: dict[str, Any]):
    """ElevenLabs TTS — high quality voice synthesis via free tier."""
    from livekit.plugins import elevenlabs

    return elevenlabs.TTS(
        model=cfg.get("model", "eleven_turbo_v2_5"),
        voice_id=cfg.get("voice_id", "bIHbv24MWmeRgasZH58o"),
        api_key=_require_env("ELEVENLABS_API_KEY"),
    )


def build_llm(cfg: dict[str, Any]):
    """Groq LLM — open source Llama model served via Groq's OpenAI-compatible API."""
    from livekit.plugins import openai as lk_openai

    return lk_openai.LLM(
        model=cfg.get("model", "llama-3.3-70b-versatile"),
        base_url=cfg.get("base_url", "https://api.groq.com/openai/v1"),
        api_key=_require_env("GROQ_API_KEY"),
    )
