"""Temporal scene graph G_t = (V_t, E_t): nodes = persons, hands, objects."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.types import SceneState
from src.interaction.hand_object.features import compute_hand_object_features

NODE_TYPES = ["person", "hand", "object"]
EDGE_TYPES = ["person-hand", "hand-object", "person-object", "object-object"]


@dataclass
class InteractionGraph:
    timestamp: float
    nodes: list[dict] = field(default_factory=list)        # {id, type, pos}
    edges: list[dict] = field(default_factory=list)        # {src, dst, type, feat}

    @classmethod
    def from_scene(cls, cur: SceneState, prev: SceneState | None = None,
                   max_obj_dist: float = 1.0) -> "InteractionGraph":
        g = cls(cur.timestamp)
        idx = {}

        def add(key, typ, pos):
            idx[key] = len(g.nodes)
            g.nodes.append({"id": key, "type": typ, "pos": np.asarray(pos, float)})

        for pid, p in cur.persons.items():
            add(("person", pid), "person", np.nanmean(p.pose_3d, 0))
            for side, j in (("left", 9), ("right", 10)):
                add(("hand", pid, side), "hand", p.pose_3d[j])
                g.edges.append({"src": idx[("person", pid)], "dst": idx[("hand", pid, side)],
                                "type": "person-hand", "feat": np.zeros(4)})
        for oid, o in cur.objects.items():
            add(("object", oid), "object", o.position_3d)
        for f in compute_hand_object_features(prev, cur):
            if f.distance <= max_obj_dist:
                g.edges.append({"src": idx[("hand", f.person_id, f.hand)], "dst": idx[("object", f.object_id)],
                                "type": "hand-object",
                                "feat": np.array([f.distance, f.approach_speed, f.relative_speed, f.object_speed])})
        oids = list(cur.objects)
        for i, a in enumerate(oids):
            for b in oids[i + 1:]:
                d = np.linalg.norm(cur.objects[a].position_3d - cur.objects[b].position_3d)
                if d <= max_obj_dist:
                    g.edges.append({"src": idx[("object", a)], "dst": idx[("object", b)],
                                    "type": "object-object", "feat": np.array([d, 0, 0, 0])})
        return g

    def to_pyg(self):
        """Convert to torch_geometric.data.Data for learned HOI models."""
        import torch
        from torch_geometric.data import Data
        x = torch.tensor(np.array([np.r_[np.nan_to_num(n["pos"]), np.eye(3)[NODE_TYPES.index(n["type"])]]
                                   for n in self.nodes]), dtype=torch.float32)
        ei = torch.tensor([[e["src"] for e in self.edges], [e["dst"] for e in self.edges]], dtype=torch.long)
        ea = torch.tensor(np.array([np.r_[e["feat"], np.eye(4)[EDGE_TYPES.index(e["type"])]]
                                    for e in self.edges]).reshape(-1, 8), dtype=torch.float32)
        return Data(x=x, edge_index=ei, edge_attr=ea)
