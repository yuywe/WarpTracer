"""Flat map collision and continuous route projection on the device."""
import warp as wp
from .lean import bicycle_step

wp.set_module_options({"enable_backward": False})


@wp.func
def map_clearance(grid: wp.array2d(dtype=float), origin: wp.vec2, cell: float, p: wp.vec3):
    fx = (p[0] - origin[0]) / cell - .5
    fy = (p[1] - origin[1]) / cell - .5
    x, y = int(wp.floor(fx + .5)), int(wp.floor(fy + .5))
    value = float(-1.0)
    if x >= 0 and y >= 0 and x < grid.shape[0] and y < grid.shape[1]:
        offset = wp.vec2(fx - float(x), fy - float(y)) * cell
        value = grid[x, y] - wp.length(offset)
    return value


@wp.kernel
def integrate_map(poses: wp.array(dtype=wp.transform), motion: wp.array(dtype=wp.vec4),
                  commands: wp.array(dtype=wp.vec4), controls: wp.array(dtype=wp.vec3),
                  collision: wp.array(dtype=int), velocities: wp.array(dtype=wp.spatial_vector),
                  contacts: wp.array(dtype=int), grid: wp.array2d(dtype=float), origin: wp.vec2, cell: float,
                  dt: float, mass: float, length: float, width: float, wheelbase: float,
                  friction: float, stiffness: float, drag: float, max_acceleration: float,
                  max_braking: float, steering_rate: float, speed_gain: float, substeps: int):
    i = wp.tid()
    pose = poses[i]
    state = motion[i]
    control = controls[i]
    command = commands[i]
    velocity = wp.spatial_vector()
    hit = int(0)
    count = int(0)
    radius = .5 * wp.sqrt(length * length + width * width)
    for _ in range(substeps):
        next_pose, state, control, velocity, unused = bicycle_step(
            pose, state, command, control, dt, mass, length, width, wheelbase,
            friction, stiffness, drag, max_acceleration, max_braking, steering_rate, speed_gain,
            0.0, 0.0, 2, wp.vec4(), 0.0)
        a, b = wp.transform_get_translation(pose), wp.transform_get_translation(next_pose)
        hit = 0
        if wp.isfinite(b[0]) and wp.isfinite(b[1]):
            # Endpoint lower bounds plus half the displacement cover the entire
            # swept collision circle, even across a thin wall in one substep.
            clearance = wp.min(map_clearance(grid, origin, cell, a), map_clearance(grid, origin, cell, b))
            if clearance <= radius + .5 * wp.length(b - a):
                hit = 1
                next_pose = pose
                q = wp.transform_get_rotation(pose)
                forward = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
                state = wp.vec4(0.0, 0.0, 0.0, wp.atan2(forward[1], forward[0]))
                velocity = wp.spatial_vector()
        pose = next_pose
        count += hit
    poses[i] = pose
    motion[i] = state
    controls[i] = control
    velocities[i] = velocity
    collision[i] = hit
    contacts[i] += count


@wp.func
def route_projection(p: wp.vec3, path: wp.array(dtype=wp.transform), arc: wp.array(dtype=wp.vec2), index: int):
    best = float(1.0e30)
    route_s = float(0.0)
    nearest = index
    n = path.shape[0]
    # 0.1 m route samples and <=0.1 m travel/control step. Keeping the search
    # local preserves route identity at nearby hairpins and bounds GPU work.
    for offset in range(-8, 9):
        j = (index + offset + n) % n
        a = wp.transform_get_translation(path[j])
        b = wp.transform_get_translation(path[(j + 1) % n])
        d = wp.vec2(b[0] - a[0], b[1] - a[1])
        v = wp.vec2(p[0] - a[0], p[1] - a[1])
        t = wp.clamp(wp.dot(v, d) / wp.dot(d, d), 0.0, 1.0)
        error = wp.length_sq(v - t * d)
        if error < best:
            best = error
            nearest = j
            route_s = arc[j][0] + t * arc[j][1]
    return route_s, nearest
