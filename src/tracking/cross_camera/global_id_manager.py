"""Persistent global identities across cameras and time.

Matching order for each new multi-view cluster:
  1. any (camera_id, local_track_id) already bound to a global id  -> reuse
  2. 3D proximity to a recently seen identity of the same class    -> reuse
  3. appearance similarity to an identity's running embedding      -> reuse
  4. otherwise allocate a new global id
"""
from __future__ import annotations

import numpy as np

from src.core.types import GlobalIdentity, Track


class GlobalIdentityManager:
    def __init__(self, start_ids: dict[str, int] | None = None, max_3d_dist: float = 0.5,
                 min_app_sim: float = 0.6, forget_after: float = 10.0, ema: float = 0.9):
        self.next_id = dict(start_ids or {"person": 1001, "object": 2001})
        self.identities: dict[int, GlobalIdentity] = {}
        self.local_to_global: dict[tuple[str, int], int] = {}
        self.max_3d, self.min_app, self.forget, self.ema = max_3d_dist, min_app_sim, forget_after, ema

    def _alloc(self, class_name: str) -> int:
        key = "person" if class_name == "person" else "object"
        gid = self.next_id[key]; self.next_id[key] += 1
        return gid

    def update(self, clusters: list[dict[str, Track]], timestamp: float,
               positions_3d: list[np.ndarray | None] | None = None) -> list[int]:
        positions_3d = positions_3d or [None] * len(clusters)
        assigned, used = [], set()
        for cluster, pos in zip(clusters, positions_3d):
            cls = next(iter(cluster.values())).class_name
            gid = self._match(cluster, cls, pos, timestamp, used)
            if gid is None:
                gid = self._alloc(cls)
                self.identities[gid] = GlobalIdentity(gid, cls)
            ident = self.identities[gid]
            for cid, t in cluster.items():
                ident.local_ids[cid] = t.local_track_id
                self.local_to_global[(cid, t.local_track_id)] = gid
                emb = t.detection.embedding if t.detection is not None else None
                if emb is not None:
                    ident.embedding = emb if ident.embedding is None else \
                        self.ema * ident.embedding + (1 - self.ema) * emb
            if pos is not None:
                ident.last_position_3d = np.asarray(pos)
            ident.last_seen = timestamp
            used.add(gid); assigned.append(gid)
        return assigned

    def _match(self, cluster, cls, pos, ts, used) -> int | None:
        votes = [self.local_to_global.get((c, t.local_track_id)) for c, t in cluster.items()]
        votes = [v for v in votes if v is not None and v not in used]
        if votes:
            return max(set(votes), key=votes.count)
        cands = [i for i in self.identities.values() if i.class_name == cls and i.global_id not in used
                 and ts - i.last_seen <= self.forget]
        if pos is not None:
            best = min(((np.linalg.norm(i.last_position_3d - pos), i.global_id) for i in cands
                        if i.last_position_3d is not None), default=(np.inf, None))
            if best[0] <= self.max_3d:
                return best[1]
        embs = [t.detection.embedding for t in cluster.values()
                if t.detection is not None and t.detection.embedding is not None]
        if embs:
            e = np.mean(embs, 0)
            best = max(((float(e @ i.embedding / (np.linalg.norm(e) * np.linalg.norm(i.embedding) + 1e-9)),
                         i.global_id) for i in cands if i.embedding is not None), default=(-1, None))
            if best[0] >= self.min_app:
                return best[1]
        return None

    def global_id_of(self, camera_id: str, local_id: int) -> int | None:
        return self.local_to_global.get((camera_id, local_id))
