from dataclasses import dataclass


@dataclass
class InterviewData:
    name: str | None = None
    prev_org: str | None = None
    prev_role: str | None = None
    exp: str | None = None
