import numpy as np

from src.calibration.coordinate_transforms import (CameraModel, camera_to_world, invert_extrinsics,
                                                   world_to_camera)
from tests.synthetic import make_rig


def test_roundtrip_world_camera():
    cam = make_rig(1)["cam_01"]
    X = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(camera_to_world(world_to_camera(X, cam.R, cam.t), cam.R, cam.t), X, atol=1e-9)


def test_invert_extrinsics_gives_center():
    cam = make_rig(3)["cam_02"]
    _, c = invert_extrinsics(cam.R, cam.t)
    np.testing.assert_allclose(c, cam.center, atol=1e-9)


def test_yaml_io(tmp_path):
    cam = make_rig(1)["cam_01"]
    cam.save(tmp_path / "c.yaml")
    c2 = CameraModel.load(tmp_path / "c.yaml")
    np.testing.assert_allclose(c2.P, cam.P)
