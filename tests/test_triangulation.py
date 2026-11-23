import numpy as np

from src.reconstruction.triangulation import (refine_nonlinear, triangulate_dlt, triangulate_ransac,
                                              triangulate_skeleton, triangulate_weighted)
from tests.synthetic import make_rig, project_skeleton, random_skeleton


def test_dlt_exact():
    cams = make_rig(3); X = np.array([0.1, -0.2, 1.2])
    pts = np.stack([c.project(X)[0] for c in cams.values()])
    np.testing.assert_allclose(triangulate_dlt([c.P for c in cams.values()], pts), X, atol=1e-6)


def test_weighted_and_nonlinear_under_noise():
    cams = make_rig(4); X = random_skeleton(1)[0]
    rng = np.random.default_rng(1)
    pts = np.stack([c.project(X)[0] for c in cams.values()]) + rng.normal(0, 1.0, (4, 2))
    P = [c.P for c in cams.values()]
    X0 = triangulate_weighted(P, pts, np.ones(4))
    X1 = refine_nonlinear(P, pts, X0)
    assert np.linalg.norm(X1 - X) < 0.01


def test_ransac_rejects_outlier_view():
    cams = make_rig(4); X = np.array([0.0, 0.0, 1.0])
    pts = np.stack([c.project(X)[0] for c in cams.values()])
    pts[2] += 200  # gross outlier
    Xr, inl = triangulate_ransac([c.P for c in cams.values()], pts)
    assert not inl[2] and np.linalg.norm(Xr - X) < 1e-4


def test_skeleton_mpjpe_small():
    cams = make_rig(3); X = random_skeleton()
    kps, conf = project_skeleton(cams, X, noise_px=0.5)
    Xh, jc = triangulate_skeleton([c.P for c in cams.values()], kps, conf)
    assert np.nanmean(np.linalg.norm(Xh - X, axis=1)) < 0.01 and (jc > 0).all()


def test_low_confidence_joint_is_nan():
    cams = make_rig(3); X = random_skeleton()
    kps, conf = project_skeleton(cams, X)
    conf[:, 5] = 0.0
    Xh, _ = triangulate_skeleton([c.P for c in cams.values()], kps, conf)
    assert np.isnan(Xh[5]).all()
