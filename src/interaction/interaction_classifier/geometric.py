"""Rule-based geometric HOI baseline with a per-(person, hand, object) state machine.

approaching -> reaching -> grasping -> holding -> moving/using -> releasing -> placing
"""
from __future__ import annotations

from collections import defaultdict

from src.core.types import InteractionEvent, SceneState
from src.interaction.hand_object.contact import contact_probability
from src.interaction.hand_object.features import compute_hand_object_features

INTERACTION_LABELS = ["none", "approaching", "reaching", "grasping", "holding", "moving",
                      "using", "releasing", "placing"]


class GeometricInteractionClassifier:
    def __init__(self, approach_dist: float = 0.6, reach_dist: float = 0.25, contact_dist: float = 0.05,
                 approach_speed: float = 0.1, move_speed: float = 0.05, hold_frames: int = 5):
        self.ad, self.rd, self.cd = approach_dist, reach_dist, contact_dist
        self.aspd, self.mspd, self.hold_frames = approach_speed, move_speed, hold_frames
        self.state: dict[tuple, str] = defaultdict(lambda: "none")
        self.contact_count: dict[tuple, int] = defaultdict(int)

    def __call__(self, prev: SceneState | None, cur: SceneState) -> list[InteractionEvent]:
        events = []
        for f in compute_hand_object_features(prev, cur):
            key = (f.person_id, f.hand, f.object_id)
            pc = contact_probability(f.distance, self.cd)
            prev_state = self.state[key]
            in_contact = pc > 0.5
            self.contact_count[key] = self.contact_count[key] + 1 if in_contact else 0

            if in_contact:
                if prev_state in ("none", "approaching", "reaching"):
                    s = "grasping"
                elif self.contact_count[key] >= self.hold_frames:
                    s = "moving" if f.object_speed > self.mspd else "holding"
                else:
                    s = prev_state
            else:
                if prev_state in ("grasping", "holding", "moving", "using"):
                    s = "placing" if f.object_speed < self.mspd else "releasing"
                elif f.distance < self.rd and f.approach_speed > 0:
                    s = "reaching"
                elif f.distance < self.ad and f.approach_speed > self.aspd:
                    s = "approaching"
                else:
                    s = "none"
            self.state[key] = s
            if s != "none":
                events.append(InteractionEvent(f.person_id, f.object_id, f.hand, s,
                                               max(pc, 0.5) if in_contact else 1 - pc, cur.timestamp))
        return events
