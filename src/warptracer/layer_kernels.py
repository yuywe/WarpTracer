"""Layered 2.5D road following and collision on the device; see layers.py."""
import warp as wp
from .lean import bicycle_step

wp.set_module_options({"enable_backward": False})


@wp.func
def surface_rotation(heading: float, gx: float, gy: float):
    """Chassis frame on a sloped surface; the 2D form of terrain.road_rotation."""
    normal = wp.normalize(wp.vec3(-gx, -gy, 1.0))
    forward = wp.normalize(wp.vec3(wp.cos(heading), wp.sin(heading), gx * wp.cos(heading) + gy * wp.sin(heading)))
    left = wp.cross(normal, forward)
    return wp.quat_from_matrix(wp.mat33(
        forward[0], left[0], normal[0],
        forward[1], left[1], normal[1],
        forward[2], left[2], normal[2]))


@wp.func
def layer_height(heights: wp.array3d(dtype=float), layer: int, origin: wp.vec2, cell: float, x: float, y: float):
    fx = (x - origin[0]) / cell - .5
    fy = (y - origin[1]) / cell - .5
    i = wp.clamp(int(wp.floor(fx)), 0, heights.shape[1] - 2)
    j = wp.clamp(int(wp.floor(fy)), 0, heights.shape[2] - 2)
    tx = wp.clamp(fx - float(i), 0.0, 1.0)
    ty = wp.clamp(fy - float(j), 0.0, 1.0)
    return ((1.0 - tx) * (1.0 - ty) * heights[layer, i, j] + tx * (1.0 - ty) * heights[layer, i + 1, j]
            + (1.0 - tx) * ty * heights[layer, i, j + 1] + tx * ty * heights[layer, i + 1, j + 1])


@wp.func
def layer_node(grid: wp.array3d(dtype=int), layer: int, origin: wp.vec2, cell: float, p: wp.vec3):
    x = int(wp.floor((p[0] - origin[0]) / cell))
    y = int(wp.floor((p[1] - origin[1]) / cell))
    value = int(0)
    if x >= 0 and y >= 0 and x < grid.shape[1] and y < grid.shape[2]:
        value = grid[layer, x, y]
    return value


@wp.func
def layer_clearance(grid: wp.array3d(dtype=float), layer: int, origin: wp.vec2, cell: float, p: wp.vec3):
    fx = (p[0] - origin[0]) / cell - .5
    fy = (p[1] - origin[1]) / cell - .5
    x, y = int(wp.floor(fx + .5)), int(wp.floor(fy + .5))
    value = float(-1.0)
    if x >= 0 and y >= 0 and x < grid.shape[1] and y < grid.shape[2]:
        offset = wp.vec2(fx - float(x), fy - float(y)) * cell
        value = grid[layer, x, y] - wp.length(offset)
    return value


@wp.func
def box_clearance(grid: wp.array3d(dtype=float), layer: int, origin: wp.vec2, cell: float,
                  p: wp.vec3, heading: float, length: float, width: float, spacing: float):
    """Minimum wall clearance over samples of the chassis outline (not a circle).

    Samples are at most `spacing` apart, so the outline between them is clear
    when every sample clears spacing / 2. Walls are areas, never thinner lines.
    """
    c, s = wp.cos(heading), wp.sin(heading)
    nx = int(wp.ceil(length / spacing))
    ny = int(wp.ceil(width / spacing))
    best = float(1.0e10)
    for k in range(2 * (nx + ny)):
        # Walk the perimeter: front, left, back, right edges.
        u, v = float(0.0), float(0.0)
        if k < nx:
            u, v = -.5 * length + length * float(k) / float(nx), -.5 * width
        elif k < nx + ny:
            u, v = .5 * length, -.5 * width + width * float(k - nx) / float(ny)
        elif k < 2 * nx + ny:
            u, v = .5 * length - length * float(k - nx - ny) / float(nx), .5 * width
        else:
            u, v = -.5 * length, .5 * width - width * float(k - 2 * nx - ny) / float(ny)
        q = wp.vec3(p[0] + c * u - s * v, p[1] + s * u + c * v, p[2])
        best = wp.min(best, layer_clearance(grid, layer, origin, cell, q))
    return best


@wp.kernel
def integrate_layers(poses: wp.array(dtype=wp.transform), motion: wp.array(dtype=wp.vec4),
                     commands: wp.array(dtype=wp.vec4), controls: wp.array(dtype=wp.vec3),
                     collision: wp.array(dtype=int), velocities: wp.array(dtype=wp.spatial_vector),
                     contacts: wp.array(dtype=int), layers: wp.array(dtype=int),
                     heights: wp.array3d(dtype=float), free: wp.array3d(dtype=int),
                     clearance: wp.array3d(dtype=float), origin: wp.vec2, cell: float,
                     ride_height: float, seam_tolerance: float,
                     dt: float, mass: float, length: float, width: float, wheelbase: float,
                     friction: float, stiffness: float, drag: float, max_acceleration: float,
                     max_braking: float, steering_rate: float, speed_gain: float, substeps: int):
    i = wp.tid()
    pose = poses[i]
    state = motion[i]
    control = controls[i]
    command = commands[i]
    layer = layers[i]
    velocity = wp.spatial_vector()
    hit = int(0)
    count = int(0)
    spacing = .05
    for _ in range(substeps):
        next_pose, state, control, velocity, unused = bicycle_step(
            pose, state, command, control, dt, mass, length, width, wheelbase,
            friction, stiffness, drag, max_acceleration, max_braking, steering_rate, speed_gain,
            0.0, 0.0, 2, wp.vec4(), 0.0)
        a, b = wp.transform_get_translation(pose), wp.transform_get_translation(next_pose)
        hit = 0
        if wp.isfinite(b[0]) and wp.isfinite(b[1]):
            # Chassis outline against this layer's walls only, padded by half the
            # substep travel. Unlike the flat map, a move that increases clearance
            # is allowed, so a car touching a wall can still steer away.
            forward = wp.quat_rotate(wp.transform_get_rotation(pose), wp.vec3(1.0, 0.0, 0.0))
            before = box_clearance(clearance, layer, origin, cell, a, wp.atan2(forward[1], forward[0]),
                                   length, width, spacing)
            after = box_clearance(clearance, layer, origin, cell, b, state[3], length, width, spacing)
            pad = .5 * spacing + .5 * wp.length(wp.vec2(b[0] - a[0], b[1] - a[1]))
            if after <= pad and after <= before:
                hit = 1
        if hit == 1:
            q = wp.transform_get_rotation(pose)
            forward = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
            state = wp.vec4(0.0, 0.0, 0.0, wp.atan2(forward[1], forward[0]))
            velocity = wp.spatial_vector()
        else:
            # Leaving this layer's road is only possible through a seam: hand
            # over to the layer that is drivable there at the same height.
            if layer_node(free, layer, origin, cell, b) == 0:
                here = layer_height(heights, layer, origin, cell, b[0], b[1])
                for other in range(heights.shape[0]):
                    if other != layer and layer_node(free, other, origin, cell, b) == 1:
                        if wp.abs(layer_height(heights, other, origin, cell, b[0], b[1]) - here) < seam_tolerance:
                            layer = other
            heading = state[3]
            gx = (layer_height(heights, layer, origin, cell, b[0] + cell, b[1])
                  - layer_height(heights, layer, origin, cell, b[0] - cell, b[1])) / (2.0 * cell)
            gy = (layer_height(heights, layer, origin, cell, b[0], b[1] + cell)
                  - layer_height(heights, layer, origin, cell, b[0], b[1] - cell)) / (2.0 * cell)
            z = layer_height(heights, layer, origin, cell, b[0], b[1]) + ride_height * wp.sqrt(1.0 + gx * gx + gy * gy)
            pose = wp.transform(wp.vec3(b[0], b[1], z), surface_rotation(heading, gx, gy))
            linear = wp.spatial_top(velocity)
            velocity = wp.spatial_vector(wp.vec3(linear[0], linear[1], (z - a[2]) / dt), wp.spatial_bottom(velocity))
        count += hit
    poses[i] = pose
    motion[i] = state
    controls[i] = control
    velocities[i] = velocity
    collision[i] = hit
    contacts[i] += count
    layers[i] = layer
