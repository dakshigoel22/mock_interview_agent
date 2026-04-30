"""Tests for InterviewData — imports directly from models.py, no LiveKit needed."""
import sys
from pathlib import Path
from dataclasses import fields

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import InterviewData


def test_interview_data_defaults():
    data = InterviewData()
    assert data.name is None
    assert data.prev_org is None
    assert data.prev_role is None
    assert data.exp is None


def test_interview_data_assignment():
    data = InterviewData()
    data.name = "Alice"
    data.prev_org = "Acme Corp"
    data.prev_role = "Engineer"
    data.exp = "2 years"
    assert data.name == "Alice"
    assert data.prev_org == "Acme Corp"
    assert data.prev_role == "Engineer"
    assert data.exp == "2 years"


def test_interview_data_field_names():
    field_names = {f.name for f in fields(InterviewData)}
    assert field_names == {"name", "prev_org", "prev_role", "exp"}


def test_interview_data_partial_fill():
    data = InterviewData(name="Bob")
    assert data.name == "Bob"
    assert data.prev_org is None
    assert data.exp is None


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
