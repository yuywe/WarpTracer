"""Export a compact pose recording as an interactive browser scene."""
from pathlib import Path

import numpy as np


def save_html(trajectory, track, vehicle, path):
    # Optional dependency: headless physics does not import or start Viser.
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        server.scene.set_up_direction("+z")
        server.initial_camera.position = (track.length, -track.width, max(track.length, track.width))
        server.initial_camera.look_at = (0.0, 0.0, 0.0)
        server.initial_camera.up = (0.0, 0.0, 1.0)
        extent_x = track.length + 2 * track.barrier_thickness
        extent_y = track.width + 2 * track.barrier_thickness
        server.scene.add_box("/floor", dimensions=(extent_x, extent_y, 0.06),
                             position=(0, 0, -0.03), color=(70, 75, 82))
        for i, (position, dimensions) in enumerate(track.barriers()):
            server.scene.add_box(f"/walls/{i}", dimensions=dimensions, position=position,
                                 color=(153, 153, 153))
        chassis = server.scene.add_box("/vehicle", dimensions=vehicle.dimensions,
                                       color=(30, 126, 238))
        scans = trajectory.lidar
        if scans is not None:
            returns = server.scene.add_point_cloud(
                "/lidar/returns", points=np.empty((0, 3), dtype=np.float32),
                colors=(255, 210, 70), point_size=0.035, point_shape="circle", precision="float32")
            rays = server.scene.add_line_segments(
                "/lidar/rays", points=np.empty((0, 2, 3), dtype=np.float32),
                colors=(60, 180, 130), line_width=1)
        pose_events = {float(t): i for i, t in enumerate(trajectory.times)}
        scan_events = {} if scans is None else {float(t): i for i, t in enumerate(scans.times)}
        serializer = server.get_scene_serializer()
        previous_time = 0.0
        for t in sorted(pose_events.keys() | scan_events.keys()):
            if t > previous_time:
                serializer.insert_sleep(t - previous_time)
            previous_time = t
            if t in pose_events:
                pose = trajectory.poses[pose_events[t]]
                chassis.position = pose[:3]
                # Warp/Newton use XYZW; Viser uses WXYZ.
                chassis.wxyz = pose[[6, 3, 4, 5]]
            if t in scan_events:
                index = scan_events[t]
                points = scans.points(index)
                # Thin only the visualization; recordings retain every beam.
                displayed = points[::max(1, (len(points) + 179) // 180)]
                displayed = displayed[np.isfinite(displayed).all(axis=1)]
                ends = points[::max(1, (len(points) + 29) // 30)]
                ends = ends[np.isfinite(ends).all(axis=1)]
                origins = np.broadcast_to(scans.poses[index, :3], ends.shape)
                returns.points = displayed.astype(np.float32)
                returns.visible = len(displayed) > 0
                rays.points = np.stack((origins, ends), axis=1).astype(np.float32)
                rays.visible = len(ends) > 0
        html = serializer.as_html(dark_mode=True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path
    finally:
        server.stop()
