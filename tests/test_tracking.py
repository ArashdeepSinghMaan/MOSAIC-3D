import numpy as np

from src.core.types import BBox, Detection, Keypoints2D, Track
from src.tracking.cross_camera import CrossCameraAssociator, GlobalIdentityManager
from src.tracking.single_camera import ByteTracker
from tests.synthetic import make_rig, random_skeleton


def test_bytetrack_keeps_ids():
    tr = ByteTracker("cam_01", min_hits=1)
    ids = []
    for f in range(10):
        dets = [Detection("person", 0.9, BBox(10 + 2 * f, 10, 60 + 2 * f, 120)),
                Detection("person", 0.9, BBox(300 - 2 * f, 20, 350 - 2 * f, 130))]
        ids.append(sorted(t.local_track_id for t in tr.update(dets)))
    assert all(i == [1, 2] for i in ids)


def _track(cam, lid, kps):
    d = Detection("person", 0.9, BBox(*kps.min(0), *kps.max(0)),
                  keypoints=Keypoints2D(kps, np.ones(len(kps))), local_track_id=lid)
    return Track(cam.camera_id, lid, "person", d.bbox, 0.9, detection=d)


def test_cross_camera_association_two_people():
    cams = make_rig(3)
    A, B = random_skeleton(seed=1, center=(-0.8, 0, 1)), random_skeleton(seed=2, center=(0.8, 0.3, 1))
    tracks = {cid: [_track(c, 1, c.project(A)), _track(c, 2, c.project(B))][:: (1 if i % 2 else -1)]
              for i, (cid, c) in enumerate(cams.items())}
    clusters = CrossCameraAssociator(cams).associate(tracks)
    assert len(clusters) == 2
    for cl in clusters:
        assert len(cl) == 3 and len({t.local_track_id for t in cl.values()}) == 1

    gm = GlobalIdentityManager()
    g1 = gm.update(clusters, 0.0)
    g2 = gm.update(clusters, 0.033)
    assert sorted(g1) == sorted(g2) == [1001, 1002]
