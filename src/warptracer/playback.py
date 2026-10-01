"""Export a compact pose recording as an interactive browser scene."""
from pathlib import Path


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
        serializer = server.get_scene_serializer()
        for i, (t, pose) in enumerate(zip(trajectory.times, trajectory.poses)):
            if i:
                serializer.insert_sleep(float(t - trajectory.times[i - 1]))
            chassis.position = pose[:3]
            # Warp/Newton use XYZW; Viser uses WXYZ.
            chassis.wxyz = pose[[6, 3, 4, 5]]
        html = serializer.as_html(dark_mode=True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path
    finally:
        server.stop()
