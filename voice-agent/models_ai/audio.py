"""Pure audio conversion helpers — no LiveKit dependency."""
from __future__ import annotations

import numpy as np


def int16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes to float32 array in [-1, 1] for Whisper."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 audio in [-1, 1] to int16 PCM bytes for LiveKit."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
