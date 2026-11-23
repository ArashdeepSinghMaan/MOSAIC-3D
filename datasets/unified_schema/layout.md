# Unified recording layout

```
<environment>/recordings/<recording_id>/
  metadata.json                 # see metadata.schema.json
  videos/cam_XX.mp4
  calibration/cam_XX.yaml       # CameraModel: K, dist, R, t (X_cam = R X_world + t, metres)
  synchronization/timestamps.json
  annotations/{detections,poses_2d,poses_3d,objects_3d,interactions,activities}.json
splits/{train,val,test}.json    # recording-level, subject-independent
```
