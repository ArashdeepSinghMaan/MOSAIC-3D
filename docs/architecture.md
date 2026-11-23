# Architecture — module boundaries

| Layer | Package | Input | Output |
|---|---|---|---|
| Acquisition | `src/acquisition` | camera URIs | `SyncedFrameSet` |
| Calibration | `src/calibration` | board images | `CameraModel` YAML |
| 2D perception | `src/perception` | image | `CameraObservation` |
| Local tracking | `src/tracking/single_camera` | detections | `Track` (local IDs) |
| Cross-camera | `src/tracking/cross_camera` | tracks per camera | clusters + `GlobalIdentity` |
| 3D reconstruction | `src/reconstruction` | clusters + cameras | `Person3D`, `Object3D` |
| Scene | `src/reconstruction/scene_fusion` | 3D entities | `SceneState` |
| HOI | `src/interaction` | `SceneState` t-1, t | `InteractionEvent` |
| Temporal | `src/temporal` | sequences | actions, segments, anticipations, event memory |

All data contracts live in `src/core/types.py`. Models are selected by name through the registries
in `src/core/registry.py`, so a config change swaps YOLO ↔ RT-DETR or ByteTrack ↔ BoT-SORT.

World convention: right-handed, metres, `X_cam = R · X_world + t`.
