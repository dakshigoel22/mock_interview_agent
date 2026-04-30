"""LiveKit TTS plugin backed by HuggingFace Inference API → Kokoro-82M (Apache 2)."""
from __future__ import annotations

import io
import os
import uuid
import wave

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

_HF_KOKORO_URL = "https://api-inference.huggingface.co/models/hexgrad/Kokoro-82M"

SAMPLE_RATE = 24_000
NUM_CHANNELS = 1


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Parse a WAV blob → (raw int16 PCM bytes, sample_rate, num_channels)."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        pcm = wf.readframes(wf.getnframes())
        return pcm, wf.getframerate(), wf.getnchannels()


class HFKokoroTTS(tts.TTS):
    """Non-streaming TTS: POSTs text to HuggingFace Kokoro-82M and plays the WAV response.

    Kokoro-82M is Apache 2 licensed and runs entirely on HuggingFace infrastructure —
    no local model download required.
    """

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        model: str = "hexgrad/Kokoro-82M",
        hf_token: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._voice = voice
        self._model = model
        self._hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._hf_url = f"https://api-inference.huggingface.co/models/{model}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "huggingface"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "HFKokoroChunkedStream":
        return HFKokoroChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class HFKokoroChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: HFKokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: HFKokoroTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import httpx

        payload = {
            "inputs": self._input_text,
            "parameters": {"voice": self._tts._voice},
        }
        headers = {"Authorization": f"Bearer {self._tts._hf_token}"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self._tts._hf_url, json=payload, headers=headers)
            response.raise_for_status()
            audio_bytes = response.content

        # HF inference API returns WAV; parse to raw PCM for LiveKit
        try:
            pcm, sample_rate, num_channels = _wav_to_pcm(audio_bytes)
        except wave.Error:
            # If not WAV, treat as raw int16 PCM at the model's native rate
            pcm, sample_rate, num_channels = audio_bytes, SAMPLE_RATE, NUM_CHANNELS

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=sample_rate,
            num_channels=num_channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()
