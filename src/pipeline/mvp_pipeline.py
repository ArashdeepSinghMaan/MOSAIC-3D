"""MVP (Section 12): multi-camera 3D human + object tracking.

Per synced frame set:
  perception (per cam) -> local tracking (per cam) -> cross-camera association
  -> global IDs -> 3D skeletons / 3D object centres -> scene -> HOI -> event memory
"""
from __future__ import annotations

from src.calibration.coordinate_transforms import CameraModel
from src.core.types import Object3D, SceneState
from src.deployment.profiler import PipelineProfiler
from src.interaction.interaction_classifier.geometric import GeometricInteractionClassifier
from src.reconstruction.object_3d.localization import localize_object_center
from src.reconstruction.pose_3d.skeleton import Skeleton3DReconstructor
from src.reconstruction.scene_fusion.scene_builder import SceneBuilder
from src.temporal.memory.event_memory import EventMemory
from src.tracking.cross_camera.association import CrossCameraAssociator
from src.tracking.cross_camera.global_id_manager import GlobalIdentityManager
from src.tracking.object_tracking.object_tracker import ObjectTracker3D
from src.tracking.single_camera.bytetrack import ByteTracker


class MVPPipeline:
    def __init__(self, cameras: dict[str, CameraModel], perception: dict, cfg: dict | None = None):
        """perception: camera_id -> PerceptionPipeline (or any callable returning CameraObservation)."""
        cfg = cfg or {}
        self.cameras, self.perception = cameras, perception
        tcfg = cfg.get("tracking", {})
        self.person_trackers = {c: ByteTracker(c, class_name="person", **tcfg.get("bytetrack", {}))
                                for c in cameras}
        self.object_trackers = {c: ByteTracker(c, **tcfg.get("bytetrack", {})) for c in cameras}
        self.associator = CrossCameraAssociator(cameras, **cfg.get("cross_camera", {}))
        self.ids = GlobalIdentityManager(**cfg.get("global_ids", {}))
        self.skel = Skeleton3DReconstructor(cameras, **cfg.get("reconstruction", {}))
        self.obj3d = ObjectTracker3D()
        self.scene = SceneBuilder()
        self.hoi = GeometricInteractionClassifier(**cfg.get("interaction", {}))
        self.memory = EventMemory()
        self.prof = PipelineProfiler()
        self.prev_scene: SceneState | None = None

    def step(self, synced) -> SceneState:
        ts = synced.timestamp
        with self.prof.section("frame_set"):
            with self.prof.section("perception"):
                obs = {c: self.perception[c](c, synced.frame_ids[c], ts, img) for c, img in synced.frames.items()}
            with self.prof.section("local_tracking"):
                p_tracks = {c: self.person_trackers[c].update(o.by_class("person")) for c, o in obs.items()}
                o_tracks = {c: self.object_trackers[c].update([d for d in o.detections if d.class_name != "person"])
                            for c, o in obs.items()}
            with self.prof.section("cross_camera"):
                p_clusters = self.associator.associate(p_tracks)
                o_clusters = self.associator.associate(o_tracks)
            with self.prof.section("reconstruction"):
                persons = []
                p_gids = self.ids.update(p_clusters, ts)
                for gid, cl in zip(p_gids, p_clusters):
                    person = self.skel(gid, cl, ts)
                    if person is not None:
                        persons.append(person)
                o_pos = [localize_object_center(self.cameras, cl) for cl in o_clusters]
                o_gids = self.ids.update(o_clusters, ts, o_pos)
                obs3d = {g: (next(iter(cl.values())).class_name, x)
                         for g, cl, x in zip(o_gids, o_clusters, o_pos) if x is not None}
                filtered = self.obj3d.step(obs3d)
                objects = [Object3D(g, obs3d[g][0], filtered[g]) for g in obs3d]
                scene = self.scene.build(ts, persons, objects)
            with self.prof.section("interaction"):
                scene.interactions = self.hoi(self.prev_scene, scene)
                self.memory.ingest(scene.interactions, ts)
        self.prev_scene = scene
        return scene
