import numpy as np

from src.core.types import Object3D, Person3D, SceneState
from src.interaction import GeometricInteractionClassifier, InteractionGraph
from src.temporal.memory import EventMemory


def _scene(t, wrist, obj):
    pose = np.zeros((17, 3)); pose[9] = wrist; pose[10] = [5, 5, 5]
    return SceneState(t, {1001: Person3D(1001, pose, np.ones(17))},
                      {2001: Object3D(2001, "screwdriver", np.asarray(obj, float))})


def test_reach_grasp_hold_place_sequence():
    clf, mem, prev, labels = GeometricInteractionClassifier(hold_frames=3), EventMemory(min_duration=0), None, []
    obj = np.array([0.5, 0, 1.0])
    traj = [obj + [0.2 - 0.04 * i, 0, 0] for i in range(6)] + [obj] * 6   # approach then contact
    for i, w in enumerate(traj):
        s = _scene(i / 30, w, obj)
        ev = clf(prev, s); mem.ingest(ev, s.timestamp); prev = s
        labels.append(next((e.label for e in ev if e.hand == "left"), "none"))
    assert "reaching" in labels and "grasping" in labels and labels[-1] == "holding"
    mem.flush()
    assert any(e.label == "grasping" for e in mem.timeline(1001))


def test_graph_construction():
    g = InteractionGraph.from_scene(_scene(0, [0.5, 0, 1.0], [0.52, 0, 1.0]))
    assert {e["type"] for e in g.edges} >= {"person-hand", "hand-object"}
