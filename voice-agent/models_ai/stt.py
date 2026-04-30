"""LiveKit STT plugin backed by Groq Whisper API (open source model, zero local setup)."""
from __future__ import annotations

import io
import os
import wave
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

if TYPE_CHECKING:
    pass

_GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _frame_to_wav(frame: rtc.AudioFrame) -> bytes:
    """Convert a LiveKit AudioFrame (int16 PCM) to a WAV byte blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(frame.num_channels)
        wf.setsampwidth(2)  # int16 = 2 bytes per sample
        wf.setframerate(frame.sample_rate)
        wf.writeframes(bytes(frame.data))
    return buf.getvalue()


class GroqWhisperSTT(stt.STT):
    """Non-streaming STT: sends a WAV clip to Groq Whisper and returns the transcript.

    The LiveKit framework's StreamAdapter (backed by VAD) wraps this into a
    streaming interface that the AgentSession requires.
    """

    def __init__(
        self,
        *,
        model: str = "whisper-large-v3-turbo",
        language: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model_name = model
        self._language = language
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "groq"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        import httpx

        frame = rtc.combine_audio_frames(buffer)
        wav_bytes = _frame_to_wav(frame)

        lang = language if isinstance(language, str) else self._language

        form: dict = {
            "model": self._model_name,
            "response_format": "json",
        }
        if lang:
            form["language"] = lang

        async with httpx.AsyncClient(timeout=conn_options.timeout or 30.0) as client:
            response = await client.post(
                _GROQ_TRANSCRIPTIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data=form,
            )
            response.raise_for_status()
            result = response.json()

        text = result.get("text", "").strip()
        detected_lang = result.get("language", lang or "en")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=detected_lang,
                    confidence=1.0,
                )
            ],
        )
