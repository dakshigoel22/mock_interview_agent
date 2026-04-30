"""Tests for InterviewData — imports directly from models.py, no LiveKit needed."""
import sys
from pathlib import Path
from dataclasses import fields

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import InterviewData


def test_interview_data_defaults():
    data = InterviewData()
    assert data.name is None
    assert data.exp is None
    assert data.experience_summary is None
    assert data.technical_notes == []


def test_interview_data_assignment():
    data = InterviewData()
    data.name = "Alice"
    data.exp = "CS grad with 2 years experience"
    data.experience_summary = "Worked on ML pipelines at a fintech."
    data.technical_notes.append({"q": "Q?", "a": "A.", "obs": "Solid."})
    assert data.name == "Alice"
    assert data.exp == "CS grad with 2 years experience"
    assert data.experience_summary == "Worked on ML pipelines at a fintech."
    assert data.technical_notes == [{"q": "Q?", "a": "A.", "obs": "Solid."}]


def test_interview_data_field_names():
    field_names = {f.name for f in fields(InterviewData)}
    assert field_names == {"name", "exp", "experience_summary", "technical_notes"}


def test_interview_data_partial_fill():
    data = InterviewData(name="Bob")
    assert data.name == "Bob"
    assert data.exp is None
    assert data.experience_summary is None
    assert data.technical_notes == []


def test_interview_data_is_mutable():
    data = InterviewData()
    data.name = "First"
    data.name = "Second"
    assert data.name == "Second"


def test_interview_data_all_fields_optional():
    # Should instantiate with no arguments
    data = InterviewData()
    assert data is not None


def test_interview_data_repr_contains_fields():
    data = InterviewData(name="Carol", exp="3 years")
    r = repr(data)
    assert "Carol" in r
    assert "3 years" in r


def test_technical_notes_independent_per_instance():
    """Regression: list default must use field(default_factory=list) so two
    InterviewData instances don't share the same list object."""
    a = InterviewData()
    b = InterviewData()
    a.technical_notes.append({"q": "x", "a": "y", "obs": "z"})
    assert b.technical_notes == []
