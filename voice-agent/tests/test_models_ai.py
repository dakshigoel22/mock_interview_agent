"""Tests for Phase 2: API-based model wrappers and factory functions.

Sections:
  A. Pure audio helpers (numpy only, no stubs)
  B. Config validation — models section keys and values
  C. GroqWhisperSTT — class behaviour (LiveKit stubbed)
  D. HFKokoroTTS — class behaviour (LiveKit stubbed)
  E. Factory functions
"""
from __future__ import annotations

import io
import sys
import types
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── helpers ───────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_wav(sample_rate: int = 24000, duration_s: float = 0.1) -> bytes:
    """Create a minimal silent WAV blob for testing."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# A. PURE TESTS — models_ai/audio.py
# ─────────────────────────────────────────────────────────────────────────────

from models_ai.audio import int16_bytes_to_float32, float32_to_int16_bytes


def test_int16_to_float32_zero():
    data = np.zeros(100, dtype=np.int16).tobytes()
    assert np.all(int16_bytes_to_float32(data) == 0.0)


def test_int16_to_float32_max_positive():
    data = np.array([32767], dtype=np.int16).tobytes()
    assert abs(int16_bytes_to_float32(data)[0] - 1.0) < 0.001


def test_int16_to_float32_max_negative():
    data = np.array([-32768], dtype=np.int16).tobytes()
    assert int16_bytes_to_float32(data)[0] < -0.999


def test_float32_to_int16_zero():
    pcm = float32_to_int16_bytes(np.zeros(100, dtype=np.float32))
    assert np.all(np.frombuffer(pcm, dtype=np.int16) == 0)


def test_float32_to_int16_clips_above_one():
    pcm = float32_to_int16_bytes(np.array([2.0, -2.0], dtype=np.float32))
    vals = np.frombuffer(pcm, dtype=np.int16)
    assert vals[0] == 32767
    assert vals[1] == -32767


def test_float32_to_int16_round_trip():
    original = np.array([0.5, -0.5, 0.25, 0.0], dtype=np.float32)
    restored = int16_bytes_to_float32(float32_to_int16_bytes(original))
    assert np.allclose(original, restored, atol=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# B. CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def test_config_llm_is_groq():
    cfg = _load_cfg()
    assert cfg["models"]["llm"]["provider"] == "groq"


def test_config_llm_model_is_llama():
    cfg = _load_cfg()
    assert "llama" in cfg["models"]["llm"]["model"].lower()


def test_config_llm_base_url_is_groq():
    cfg = _load_cfg()
    assert "groq.com" in cfg["models"]["llm"]["base_url"]


def test_config_stt_provider_is_groq_whisper():
    cfg = _load_cfg()
    assert "groq" in cfg["models"]["stt"]["provider"]


def test_config_stt_model_is_whisper():
    cfg = _load_cfg()
    assert "whisper" in cfg["models"]["stt"]["model"].lower()


def test_config_tts_provider_is_hf_kokoro():
    cfg = _load_cfg()
    assert "kokoro" in cfg["models"]["tts"]["provider"].lower()


def test_config_tts_has_voice():
    cfg = _load_cfg()
    voice = cfg["models"]["tts"]["voice"]
    assert isinstance(voice, str) and len(voice) > 0


def test_config_tts_model_is_kokoro():
    cfg = _load_cfg()
    assert "kokoro" in cfg["models"]["tts"]["model"].lower()


def test_config_silence_timeout_positive():
    cfg = _load_cfg()
    assert cfg["interview"]["silence_timeout_seconds"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# C. GroqWhisperSTT class tests (LiveKit stubbed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_livekit_stubs():
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = MagicMock()
    sys.modules.setdefault("dotenv", dotenv_mod)

    livekit = types.ModuleType("livekit")
    rtc_mod = types.ModuleType("livekit.rtc")
    rtc_mod.EventEmitter = object
    rtc_mod.AudioFrame = MagicMock()
    mock_frame = MagicMock()
    mock_frame.data = b"\x00\x00" * 480
    mock_frame.num_channels = 1
    mock_frame.sample_rate = 16000
    rtc_mod.combine_audio_frames = MagicMock(return_value=mock_frame)
    livekit.rtc = rtc_mod

    agents = types.ModuleType("livekit.agents")
    stt_mod = types.ModuleType("livekit.agents.stt")

    class _STTCapabilities:
        def __init__(self, streaming=False, interim_results=False, **kw):
            self.streaming = streaming

    class _STT:
        def __init__(self, *, capabilities, **kw):
            self._capabilities = capabilities
            self._label = "test.STT"

    class _SpeechEventType:
        FINAL_TRANSCRIPT = "final_transcript"

    class _SpeechData:
        def __init__(self, text="", language="en", confidence=1.0, **kw):
            self.text = text
            self.language = language

    class _SpeechEvent:
        def __init__(self, type, alternatives=None, **kw):
            self.type = type
            self.alternatives = alternatives or []

    stt_mod.STT = _STT
    stt_mod.STTCapabilities = _STTCapabilities
    stt_mod.SpeechEventType = _SpeechEventType
    stt_mod.SpeechData = _SpeechData
    stt_mod.SpeechEvent = _SpeechEvent

    tts_mod = types.ModuleType("livekit.agents.tts")

    class _TTSCapabilities:
        def __init__(self, streaming=False, **kw):
            pass

    class _TTS:
        def __init__(self, *, capabilities, sample_rate, num_channels, **kw):
            self._sample_rate = sample_rate
            self._num_channels = num_channels
            self._label = "test.TTS"

    class _ChunkedStream:
        def __init__(self, *, tts, input_text, conn_options, **kw):
            self._tts = tts
            self._input_text = input_text

    tts_mod.TTS = _TTS
    tts_mod.TTSCapabilities = _TTSCapabilities
    tts_mod.ChunkedStream = _ChunkedStream
    tts_mod.AudioEmitter = MagicMock

    types_mod = types.ModuleType("livekit.agents.types")
    sentinel = object()
    types_mod.NOT_GIVEN = sentinel
    types_mod.DEFAULT_API_CONNECT_OPTIONS = MagicMock()
    types_mod.DEFAULT_API_CONNECT_OPTIONS.timeout = 30.0
    types_mod.APIConnectOptions = MagicMock
    types_mod.NotGivenOr = object

    utils_mod = types.ModuleType("livekit.agents.utils")
    utils_mod.AudioBuffer = list

    agents.stt = stt_mod
    agents.tts = tts_mod

    return {
        "dotenv": dotenv_mod,
        "livekit": livekit,
        "livekit.rtc": rtc_mod,
        "livekit.agents": agents,
        "livekit.agents.stt": stt_mod,
        "livekit.agents.tts": tts_mod,
        "livekit.agents.types": types_mod,
        "livekit.agents.utils": utils_mod,
        "livekit.plugins": MagicMock(),
        "livekit.plugins.openai": MagicMock(),
        "livekit.plugins.silero": MagicMock(),
        "livekit.plugins.noise_cancellation": MagicMock(),
    }


@pytest.fixture(scope="module")
def lk_stubs():
    stubs = _make_livekit_stubs()
    with patch.dict(sys.modules, stubs):
        for m in ("models_ai.stt", "models_ai.tts"):
            sys.modules.pop(m, None)
        yield stubs


def test_groq_whisper_stores_model(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.stt import GroqWhisperSTT
        s = GroqWhisperSTT(model="whisper-large-v3-turbo", api_key="test-key")
        assert s._model_name == "whisper-large-v3-turbo"
        assert s._api_key == "test-key"


def test_groq_whisper_provider_property(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.stt import GroqWhisperSTT
        assert GroqWhisperSTT(api_key="k").provider == "groq"


def test_groq_whisper_model_property(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.stt import GroqWhisperSTT
        s = GroqWhisperSTT(model="whisper-large-v3-turbo", api_key="k")
        assert s.model == "whisper-large-v3-turbo"


def test_groq_whisper_no_api_key_reads_env(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}):
            sys.modules.pop("models_ai.stt", None)
            from models_ai.stt import GroqWhisperSTT
            assert GroqWhisperSTT()._api_key == "env-key"


def test_frame_to_wav_produces_valid_wav(lk_stubs):
    """_frame_to_wav should produce parseable WAV bytes."""
    with patch.dict(sys.modules, lk_stubs):
        sys.modules.pop("models_ai.stt", None)
        from models_ai.stt import _frame_to_wav, GroqWhisperSTT

        mock_frame = MagicMock()
        mock_frame.num_channels = 1
        mock_frame.sample_rate = 16000
        mock_frame.data = b"\x00\x00" * 160  # 160 int16 samples = 10ms at 16kHz

        wav = _frame_to_wav(mock_frame)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getnframes() == 160


async def test_groq_whisper_recognize_calls_api(lk_stubs):
    """_recognize_impl should POST to Groq and return FINAL_TRANSCRIPT."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "Hello world", "language": "en"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict(sys.modules, {**lk_stubs, "httpx": fake_httpx}):
        sys.modules.pop("models_ai.stt", None)
        from models_ai.stt import GroqWhisperSTT

        s = GroqWhisperSTT(model="whisper-large-v3-turbo", api_key="test-key")
        conn_opts = MagicMock()
        conn_opts.timeout = 30.0
        event = await s._recognize_impl([], conn_options=conn_opts)

    assert event.type == "final_transcript"
    assert event.alternatives[0].text == "Hello world"


# ─────────────────────────────────────────────────────────────────────────────
# D. HFKokoroTTS class tests
# ─────────────────────────────────────────────────────────────────────────────

def test_hf_kokoro_stores_params(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.tts import HFKokoroTTS
        t = HFKokoroTTS(voice="af_heart", model="hexgrad/Kokoro-82M", hf_token="tok")
        assert t._voice == "af_heart"
        assert t._model == "hexgrad/Kokoro-82M"
        assert t._hf_token == "tok"


def test_hf_kokoro_provider_property(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.tts import HFKokoroTTS
        assert HFKokoroTTS(hf_token="t").provider == "huggingface"


def test_hf_kokoro_model_property(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.tts import HFKokoroTTS
        t = HFKokoroTTS(model="hexgrad/Kokoro-82M", hf_token="t")
        assert t.model == "hexgrad/Kokoro-82M"


def test_hf_kokoro_no_token_reads_env(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        with patch.dict("os.environ", {"HF_TOKEN": "env-tok"}):
            sys.modules.pop("models_ai.tts", None)
            from models_ai.tts import HFKokoroTTS
            assert HFKokoroTTS()._hf_token == "env-tok"


def test_hf_kokoro_synthesize_returns_chunked_stream(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        from models_ai.tts import HFKokoroTTS, HFKokoroChunkedStream
        t = HFKokoroTTS(hf_token="tok")
        stream = t.synthesize("Hello")
        assert isinstance(stream, HFKokoroChunkedStream)
        assert stream._input_text == "Hello"


def test_wav_to_pcm_round_trip():
    """_wav_to_pcm should correctly extract PCM from a WAV blob."""
    with patch.dict(sys.modules, _make_livekit_stubs()):
        sys.modules.pop("models_ai.tts", None)
        from models_ai.tts import _wav_to_pcm

        wav = _make_wav(sample_rate=24000, duration_s=0.1)
        pcm, sr, ch = _wav_to_pcm(wav)
        assert sr == 24000
        assert ch == 1
        assert len(pcm) == int(24000 * 0.1) * 2  # 16-bit samples


# ─────────────────────────────────────────────────────────────────────────────
# E. Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def test_build_stt_creates_groq_whisper(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        sys.modules.pop("models_ai", None)
        sys.modules.pop("models_ai.stt", None)
        with patch.dict("os.environ", {"GROQ_API_KEY": "test"}):
            from models_ai import build_stt
            from models_ai.stt import GroqWhisperSTT
            s = build_stt({"model": "whisper-large-v3-turbo"})
            assert isinstance(s, GroqWhisperSTT)
            assert s._model_name == "whisper-large-v3-turbo"


def test_build_tts_creates_hf_kokoro(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        sys.modules.pop("models_ai", None)
        sys.modules.pop("models_ai.tts", None)
        with patch.dict("os.environ", {"HF_TOKEN": "test"}):
            from models_ai import build_tts
            from models_ai.tts import HFKokoroTTS
            t = build_tts({"voice": "af_sky", "model": "hexgrad/Kokoro-82M"})
            assert isinstance(t, HFKokoroTTS)
            assert t._voice == "af_sky"


def test_build_tts_defaults(lk_stubs):
    with patch.dict(sys.modules, lk_stubs):
        sys.modules.pop("models_ai", None)
        sys.modules.pop("models_ai.tts", None)
        with patch.dict("os.environ", {"HF_TOKEN": "test"}):
            from models_ai import build_tts
            t = build_tts({})
            assert t._voice == "af_heart"
            assert "Kokoro" in t._model
