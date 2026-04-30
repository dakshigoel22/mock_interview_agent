"""Tests for config.yaml loading and structure."""
import pytest
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_config_file_exists():
    assert CONFIG_PATH.exists(), "config.yaml must exist"


def test_persona_keys(cfg):
    required = {"name", "title", "company", "interviewing_for"}
    assert required.issubset(cfg["persona"].keys()), f"Missing keys: {required - cfg['persona'].keys()}"


def test_persona_values_non_empty(cfg):
    for key, value in cfg["persona"].items():
        assert value and str(value).strip(), f"persona.{key} must not be empty"


def test_models_keys(cfg):
    required = {"llm", "stt", "tts"}
    assert required.issubset(cfg["models"].keys()), f"Missing keys: {required - cfg['models'].keys()}"


def test_models_values_non_empty(cfg):
    for key, value in cfg["models"].items():
        assert value and str(value).strip(), f"models.{key} must not be empty"


def test_interview_silence_timeout(cfg):
    timeout = cfg["interview"]["silence_timeout_seconds"]
    assert isinstance(timeout, int), "silence_timeout_seconds must be an integer"
    assert timeout > 0, "silence_timeout_seconds must be positive"


def test_llm_is_dict_with_model(cfg):
    llm = cfg["models"]["llm"]
    assert isinstance(llm, dict), "models.llm should be a dict (provider config)"
    assert "model" in llm and llm["model"], "models.llm.model must be non-empty"


def test_stt_is_dict_with_provider(cfg):
    stt = cfg["models"]["stt"]
    assert isinstance(stt, dict), "models.stt should be a dict (provider config)"
    assert "provider" in stt and stt["provider"], "models.stt.provider must be non-empty"


def test_tts_is_dict_with_provider(cfg):
    tts = cfg["models"]["tts"]
    assert isinstance(tts, dict), "models.tts should be a dict (provider config)"
    assert "provider" in tts and tts["provider"], "models.tts.provider must be non-empty"
