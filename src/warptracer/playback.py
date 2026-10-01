"""Export a compact pose recording as an interactive browser scene."""
from pathlib import Path

import numpy as np


def save_html(trajectory, track, vehicle, path):
    # Optional dependency: headless physics does not import or start Viser.
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    try:
        server.scene.set_up_direction("+z")
        server.initial_camera.position = (8.0, -10.0, 9.0)
        server.initial_camera.look_at = (0.0, -0.5, 0.0)
        server.initial_camera.up = (0.0, 0.0, 1.0)
        extent_x = track.straight_length + 2 * track.bend_radius + track.lane_width + 2
        extent_y = 2 * track.bend_radius + track.lane_width + 2
        server.scene.add_box("/floor", dimensions=(extent_x, extent_y, 0.06),
                             position=(0, 0, -0.03), color=(52, 87, 70))
        vertices, faces = track.road_mesh()
        server.scene.add_mesh_simple("/road", vertices, faces, color=(53, 59, 69), side="double")
        for i, (position, dimensions, yaw, color) in enumerate(track.barriers()):
            server.scene.add_box(f"/barriers/{i}", dimensions=dimensions, position=position,
                                 wxyz=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)),
                                 color=tuple(round(255 * c) for c in color))
        loop = track.loop()
        points = np.column_stack((loop, np.full(len(loop), 0.006)))
        server.scene.add_line_segments("/centerline", points=np.stack((points, np.roll(points, -1, axis=0)), axis=1),
                                       colors=(217, 181, 78), line_width=2)
        # The parent carries the only dynamic transform. All detail below is visual.
        chassis = server.scene.add_frame("/vehicle", show_axes=False)
        server.scene.add_box("/vehicle/chassis", dimensions=vehicle.dimensions, color=(30, 126, 238))
        server.scene.add_box("/vehicle/deck", dimensions=(vehicle.length * .42, vehicle.width * .72, .045),
                             position=(-.035, 0, vehicle.height / 2 + .0225), color=(22, 32, 49))
        server.scene.add_box("/vehicle/front_marker", dimensions=(.05, vehicle.width * .85, .008),
                             position=(vehicle.length * .35, 0, vehicle.height / 2 + .004), color=(247, 183, 51))
        # Simple black side blocks suggest tires without implying wheel dynamics.
        for x in (-vehicle.length * .30, vehicle.length * .30):
            for y in (-vehicle.width / 2, vehicle.width / 2):
                server.scene.add_box(f"/vehicle/tire_{x}_{y}", dimensions=(.095, .04, .08),
                                     position=(x, y, -.01), color=(20, 23, 28))
        server.scene.add_frame("/world_axes", axes_length=.5, axes_radius=.006,
                               position=(0, 0, .01))
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
