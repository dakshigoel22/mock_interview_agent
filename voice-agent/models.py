from dataclasses import dataclass, field


@dataclass
class InterviewData:
    name: str | None = None
    exp: str | None = None
    experience_summary: str | None = None
    technical_notes: list[dict] = field(default_factory=list)
