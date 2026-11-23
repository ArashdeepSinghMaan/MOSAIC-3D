"""3D scene visualisation with Open3D (skeletons, hands, objects, cameras, trajectories)."""
from __future__ import annotations

import numpy as np

from src.calibration.coordinate_transforms import CameraModel
from src.core.types import SceneState
from src.perception.human_pose.base import COCO17_SKELETON


def _lineset(points, edges, color):
    import open3d as o3d
    pts = np.nan_to_num(points)
    edges = [(a, b) for a, b in edges if not (np.isnan(points[a]).any() or np.isnan(points[b]).any())]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(edges))
    ls.paint_uniform_color(color)
    return ls


def scene_geometries(scene: SceneState, cameras: dict[str, CameraModel] | None = None) -> list:
    import open3d as o3d
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.3)]
    for gid, p in scene.persons.items():
        rng = np.random.default_rng(gid)
        geoms.append(_lineset(p.pose_3d, COCO17_SKELETON, rng.random(3)))
    for o in scene.objects.values():
        s = o3d.geometry.TriangleMesh.create_sphere(0.03); s.translate(o.position_3d)
        s.paint_uniform_color([1, .5, 0]); geoms.append(s)
    for cam in (cameras or {}).values():
        w, h = cam.image_size
        geoms.append(o3d.geometry.LineSet.create_camera_visualization(
            w, h, cam.K, np.vstack([np.c_[cam.R, cam.t], [0, 0, 0, 1]]), 0.2))
    return geoms


def show_scene(scene: SceneState, cameras=None) -> None:
    import open3d as o3d
    o3d.visualization.draw_geometries(scene_geometries(scene, cameras))
