"""Long-horizon event memory: collapse frame-level interactions into a queryable timeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from src.core.types import InteractionEvent

_TO_EVENT = {"grasping": "Picked up", "placing": "Placed", "releasing": "Released",
             "using": "Used", "moving": "Moved"}


@dataclass
class Event:
    person_id: int
    label: str
    object_id: int | None
    start: float
    end: float
    meta: dict = field(default_factory=dict)


class EventMemory:
    def __init__(self, min_duration: float = 0.2, merge_gap: float = 0.5,
                 session_start: datetime | None = None):
        self.events: list[Event] = []
        self._open: dict[tuple, Event] = {}
        self.min_duration, self.merge_gap = min_duration, merge_gap
        self.session_start = session_start

    def ingest(self, interactions: list[InteractionEvent], timestamp: float) -> None:
        seen = set()
        for e in interactions:
            key = (e.person_id, e.object_id, e.label)
            seen.add(key)
            if key in self._open:
                self._open[key].end = timestamp
            else:
                self._open[key] = Event(e.person_id, e.label, e.object_id, timestamp, timestamp,
                                        {"hand": e.hand})
        for key in [k for k in self._open if k not in seen]:
            ev = self._open[key]
            if timestamp - ev.end > self.merge_gap:
                if ev.end - ev.start >= self.min_duration or ev.label in ("grasping", "placing", "releasing"):
                    self.events.append(ev)
                del self._open[key]

    def flush(self) -> None:
        self.events.extend(self._open.values())
        self._open.clear()
        self.events.sort(key=lambda e: e.start)

    def timeline(self, person_id: int | None = None) -> list[Event]:
        return [e for e in sorted(self.events, key=lambda e: e.start)
                if person_id is None or e.person_id == person_id]

    def query(self, label: str | None = None, object_id: int | None = None,
              t0: float = float("-inf"), t1: float = float("inf")) -> list[Event]:
        return [e for e in self.events if (label is None or e.label == label)
                and (object_id is None or e.object_id == object_id) and e.start >= t0 and e.end <= t1]

    def render(self, person_id: int) -> str:
        lines = [f"Worker {person_id}:", ""]
        for e in self.timeline(person_id):
            if e.label not in _TO_EVENT:
                continue
            ts = ((self.session_start + timedelta(seconds=e.start)).strftime("%H:%M:%S")
                  if self.session_start else f"{e.start:8.2f}s")
            lines.append(f"{ts} - {_TO_EVENT[e.label]} object {e.object_id}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.timeline()], f, indent=1)
