"""Bicycle dynamics with optional oval road following, shared across kernels."""
import warp as wp
from .terrain import road_height, road_gradient, road_rotation


class LeanState:
    def __init__(self, device, count=1):
        self.body_q = wp.empty(count, dtype=wp.transform, device=device)
        self.body_qd = wp.empty(count, dtype=wp.spatial_vector, device=device)


@wp.func
def ellipse_box_distance(x: float, y: float, heading: float, length: float, width: float,
                         a: float, b: float):
    """Minimum normalized ellipse radius anywhere in the chassis rectangle."""
    c, s = wp.cos(heading), wp.sin(heading)
    if wp.abs(x * c + y * s) <= length / 2.0 and wp.abs(-x * s + y * c) <= width / 2.0:
        return float(0.0)
    p = wp.vec2(x / a, y / b)
    f = wp.vec2(c * length / (2.0 * a), s * length / (2.0 * b))
    l = wp.vec2(-s * width / (2.0 * a), c * width / (2.0 * b))
    best = float(1.0e10)
    for edge in range(4):
        start, end = p - f - l, p + f - l
        if edge == 1:
            start, end = p + f - l, p + f + l
        if edge == 2:
            start, end = p + f + l, p - f + l
        if edge == 3:
            start, end = p - f + l, p - f - l
        segment = end - start
        t = wp.clamp(-wp.dot(start, segment) / wp.dot(segment, segment), 0.0, 1.0)
        best = wp.min(best, wp.length(start + t * segment))
    return best


@wp.func
def bicycle_step(pose: wp.transform, state: wp.vec4, command: wp.vec4, control: wp.vec3,
    dt: float, mass: float, length: float, width: float, wheelbase: float,
    friction: float, stiffness: float, drag: float, max_acceleration: float,
    max_braking: float, steering_rate: float, speed_gain: float,
    track_length: float, track_width: float, oval: int, road: wp.vec4, wall_padding: float,
):
    u, v, yaw_rate, heading = state[0], state[1], state[2], state[3]
    throttle, brake = command[0], command[1]
    if command[3] >= 0.0:
        acceleration = speed_gain * (command[3] - u)
        throttle = wp.clamp(acceleration / max_acceleration, 0.0, 1.0)
        brake = wp.clamp(-acceleration / max_braking, 0.0, 1.0)
        if command[3] == 0.0:
            throttle, brake = 0.0, 1.0
    if brake > 0.0:
        throttle = 0.0
    steering = control[2] + wp.clamp(command[2] - control[2],
                                         -steering_rate * dt, steering_rate * dt)
    control = wp.vec3(throttle, brake, steering)
    c, s = wp.cos(steering), wp.sin(steering)
    axle = 0.5 * wheelbase
    inertia = mass * (length * length + width * width) / 12.0
    load = friction * mass * 9.81 * 0.5
    braking = mass * max_braking * brake * wp.clamp(u / 0.2, -1.0, 1.0)
    stopping = mass * wp.abs(u) / dt
    braking = wp.clamp(braking, -stopping, stopping)
    longitudinal = wp.clamp(mass * max_acceleration * throttle - braking, -2.0 * load, 2.0 * load)
    # Each axle uses half the longitudinal force; its remaining budget is lateral.
    lateral_limit = wp.sqrt(wp.max(0.0, load * load - 0.25 * longitudinal * longitudinal))
    # Implicit lateral/yaw solve prevents low-speed stiffness from destabilizing dt.
    k = 2.0 * stiffness
    h = dt * k
    a11 = 1.0 + h * (c * c + 1.0) / mass
    a12 = dt * u + h * axle * (c * c - 1.0) / mass
    a21 = h * axle * (c * c - 1.0) / inertia
    a22 = 1.0 + h * axle * axle * (c * c + 1.0) / inertia
    b1 = v + h * u * s * c / mass
    b2 = yaw_rate + h * axle * u * s * c / inertia
    det = a11 * a22 - a12 * a21
    predicted_v = (b1 * a22 - b2 * a12) / det
    predicted_yaw = (b2 * a11 - b1 * a21) / det
    front = wp.clamp(k * (u * s - (predicted_v + axle * predicted_yaw) * c), -lateral_limit, lateral_limit)
    rear = wp.clamp(-k * (predicted_v - axle * predicted_yaw), -lateral_limit, lateral_limit)
    next_yaw = yaw_rate + dt * axle * (front * c - rear) / inertia
    next_v = v + dt * ((front * c + rear) / mass - next_yaw * u)
    next_u = u + dt * ((longitudinal - front * s - 4.0 * drag * u) / mass + next_yaw * next_v)
    position = wp.transform_get_translation(pose)
    x = position[0] + dt * (wp.cos(heading) * next_u - wp.sin(heading) * next_v)
    y = position[1] + dt * (wp.sin(heading) * next_u + wp.cos(heading) * next_v)
    heading += dt * next_yaw
    heading = wp.atan2(wp.sin(heading), wp.cos(heading))
    hit_wall = 0
    if oval != 0:
        # Test the whole padded rectangle against both ellipses. Vertex checks
        # suffice for the convex outer boundary; inner checks include all edges.
        padded_length, padded_width = length + 2.0 * wall_padding, width + 2.0 * wall_padding
        a, b = .5 * track_length, .5 * track_width
        center = wp.vec2(x / a, y / b)
        f = wp.vec2(wp.cos(heading) * padded_length / (2.0 * a), wp.sin(heading) * padded_length / (2.0 * b))
        l = wp.vec2(-wp.sin(heading) * padded_width / (2.0 * a), wp.cos(heading) * padded_width / (2.0 * b))
        limit = 1.0 - wall_padding / b
        scale = float(1.0)
        aa = wp.dot(center, center)
        if aa > 1.0e-10:
            for corner in range(4):
                offset = f + l
                if corner == 1:
                    offset = f - l
                if corner == 2:
                    offset = -f - l
                if corner == 3:
                    offset = -f + l
                bb = 2.0 * wp.dot(center, offset)
                cc = wp.dot(offset, offset) - limit * limit
                root = (-bb + wp.sqrt(wp.max(0.0, bb * bb - 4.0 * aa * cc))) / (2.0 * aa)
                scale = wp.min(scale, root)
        if scale < 1.0:
            x, y = x * scale, y * scale
            hit_wall = 1
        a, b = a - road[0], b - road[0]
        inner = wp.sqrt(x * x / (a * a) + y * y / (b * b))
        if ellipse_box_distance(x, y, heading, padded_length, padded_width, a, b) <= 1.0:
            if inner > 1.0e-8:
                radius = .5 * wp.sqrt(padded_length * padded_length + padded_width * padded_width)
                lo, hi = float(1.0), (1.0 + radius / b) / inner
                for _ in range(12):
                    mid = .5 * (lo + hi)
                    if ellipse_box_distance(x * mid, y * mid, heading, padded_length, padded_width, a, b) <= 1.0:
                        lo = mid
                    else:
                        hi = mid
                x, y = x * hi, y * hi
            else:
                x, y = 0.0, -b - padded_length
            hit_wall = 1
    else:
        # Exact axis extents of the oriented box inside a rectangular enclosure.
        hx = 0.5 * (wp.abs(wp.cos(heading)) * length + wp.abs(wp.sin(heading)) * width)
        hy = 0.5 * (wp.abs(wp.sin(heading)) * length + wp.abs(wp.cos(heading)) * width)
        limit_x, limit_y = 0.5 * track_length - hx, 0.5 * track_width - hy
        if wp.abs(x) >= limit_x or wp.abs(y) >= limit_y:
            x = wp.clamp(x, -limit_x, limit_x)
            y = wp.clamp(y, -limit_y, limit_y)
            hit_wall = 1
    if hit_wall != 0:
        next_u, next_v, next_yaw = 0.0, 0.0, 0.0
    state = wp.vec4(next_u, next_v, next_yaw, heading)
    out_pose = wp.transform(wp.vec3(x, y, position[2]), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), heading))
    velocity = wp.spatial_vector(
        wp.vec3(wp.cos(heading) * next_u - wp.sin(heading) * next_v,
                wp.sin(heading) * next_u + wp.cos(heading) * next_v, 0.0),
        wp.vec3(0.0, 0.0, next_yaw))
    if oval != 0:
        gradient = road_gradient(x, road[1], road[2])
        z = road_height(x, road[1], road[2]) + road[3] * wp.sqrt(1.0 + gradient * gradient)
        rotation = road_rotation(heading, gradient)
        out_pose = wp.transform(wp.vec3(x, y, z), rotation)
        # Report actual world Z motion and angular velocity of the surface frame.
        delta = rotation * wp.quat_inverse(wp.transform_get_rotation(pose))
        if delta[3] < 0.0:
            delta = -delta
        axis = wp.vec3(delta[0], delta[1], delta[2])
        angular = wp.vec3(0.0, 0.0, 0.0)
        norm = wp.length(axis)
        if norm > 1.0e-8:
            angular = axis * (2.0 * wp.atan2(norm, delta[3]) / (norm * dt))
        velocity = wp.spatial_vector(wp.vec3(velocity[0], velocity[1], (z - position[2]) / dt), angular)
        if hit_wall != 0:
            velocity = wp.spatial_vector()
    return out_pose, state, control, velocity, hit_wall


@wp.kernel
def integrate_bicycle(
    poses: wp.array(dtype=wp.transform),
    motion: wp.array(dtype=wp.vec4), commands: wp.array(dtype=wp.vec4),
    controls: wp.array(dtype=wp.vec3), collision: wp.array(dtype=int),
    out_poses: wp.array(dtype=wp.transform), out_velocities: wp.array(dtype=wp.spatial_vector),
    contact_counts: wp.array(dtype=int),
    dt: float, mass: float, length: float, width: float, wheelbase: float,
    friction: float, stiffness: float, drag: float, max_acceleration: float,
    max_braking: float, steering_rate: float, speed_gain: float,
    track_length: float, track_width: float, oval: int, road: wp.vec4, wall_padding: float,
):
    i = wp.tid()
    pose, state, control, velocity, hit_wall = bicycle_step(
        poses[i], motion[i], commands[i], controls[i],
        dt, mass, length, width, wheelbase, friction, stiffness, drag, max_acceleration, max_braking, steering_rate, speed_gain, track_length, track_width, oval, road, wall_padding)
    out_poses[i] = pose
    motion[i] = state
    controls[i] = control
    out_velocities[i] = velocity
    collision[i] = hit_wall
    contact_counts[i] += hit_wall


@wp.kernel
def integrate_bicycle_fused(
    poses: wp.array(dtype=wp.transform),
    motion: wp.array(dtype=wp.vec4), commands: wp.array(dtype=wp.vec4),
    controls: wp.array(dtype=wp.vec3), collision: wp.array(dtype=int),
    out_poses: wp.array(dtype=wp.transform), out_velocities: wp.array(dtype=wp.spatial_vector),
    contact_counts: wp.array(dtype=int),
    dt: float, mass: float, length: float, width: float, wheelbase: float,
    friction: float, stiffness: float, drag: float, max_acceleration: float,
    max_braking: float, steering_rate: float, speed_gain: float,
    track_length: float, track_width: float, oval: int, road: wp.vec4, wall_padding: float,
    substeps: int,
):
    i = wp.tid()
    pose = poses[i]
    state = motion[i]
    control = controls[i]
    command = commands[i]
    velocity = wp.spatial_vector()
    hit_wall = int(0)
    contacts = int(0)
    # Sequential integration stays local to this car; the physics dt is unchanged.
    for _ in range(substeps):
        pose, state, control, velocity, hit_wall = bicycle_step(
            pose, state, command, control,
            dt, mass, length, width, wheelbase, friction, stiffness, drag, max_acceleration, max_braking, steering_rate, speed_gain, track_length, track_width, oval, road, wall_padding)
        contacts += hit_wall
    out_poses[i] = pose
    motion[i] = state
    controls[i] = control
    out_velocities[i] = velocity
    collision[i] = hit_wall
    contact_counts[i] += contacts
