"""Flat-ground dynamic bicycle with bounded grip and stop-on-wall behavior.

One kernel advances planar velocities and yaw. Height is fixed; the pose remains
3D so the existing mounted ray caster and playback need no special conversion.
"""
import warp as wp


class LeanState:
    def __init__(self, device):
        self.body_q = wp.empty(1, dtype=wp.transform, device=device)
        self.body_qd = wp.empty(1, dtype=wp.spatial_vector, device=device)


@wp.kernel
def integrate_bicycle(
    poses: wp.array(dtype=wp.transform),
    motion: wp.array(dtype=wp.vec4), commands: wp.array(dtype=wp.vec4),
    controls: wp.array(dtype=wp.vec3), collision: wp.array(dtype=int),
    out_poses: wp.array(dtype=wp.transform), out_velocities: wp.array(dtype=wp.spatial_vector),
    dt: float, mass: float, length: float, width: float, wheelbase: float,
    friction: float, stiffness: float, drag: float, max_acceleration: float,
    max_braking: float, steering_rate: float, speed_gain: float,
    track_length: float, track_width: float,
):
    state = motion[0]
    u, v, yaw_rate, heading = state[0], state[1], state[2], state[3]
    command = commands[0]
    throttle, brake = command[0], command[1]
    if command[3] >= 0.0:
        acceleration = speed_gain * (command[3] - u)
        throttle = wp.clamp(acceleration / max_acceleration, 0.0, 1.0)
        brake = wp.clamp(-acceleration / max_braking, 0.0, 1.0)
        if command[3] == 0.0:
            throttle, brake = 0.0, 1.0
    if brake > 0.0:
        throttle = 0.0
    steering = controls[0][2] + wp.clamp(command[2] - controls[0][2],
                                         -steering_rate * dt, steering_rate * dt)
    controls[0] = wp.vec3(throttle, brake, steering)
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
    position = wp.transform_get_translation(poses[0])
    x = position[0] + dt * (wp.cos(heading) * next_u - wp.sin(heading) * next_v)
    y = position[1] + dt * (wp.sin(heading) * next_u + wp.cos(heading) * next_v)
    heading += dt * next_yaw
    heading = wp.atan2(wp.sin(heading), wp.cos(heading))
    # Exact axis extents of the oriented box inside a rectangular enclosure.
    hx = 0.5 * (wp.abs(wp.cos(heading)) * length + wp.abs(wp.sin(heading)) * width)
    hy = 0.5 * (wp.abs(wp.sin(heading)) * length + wp.abs(wp.cos(heading)) * width)
    limit_x, limit_y = 0.5 * track_length - hx, 0.5 * track_width - hy
    collision[0] = 0
    if wp.abs(x) >= limit_x or wp.abs(y) >= limit_y:
        x = wp.clamp(x, -limit_x, limit_x)
        y = wp.clamp(y, -limit_y, limit_y)
        next_u, next_v, next_yaw = 0.0, 0.0, 0.0
        collision[0] = 1
    motion[0] = wp.vec4(next_u, next_v, next_yaw, heading)
    out_poses[0] = wp.transform(wp.vec3(x, y, position[2]), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), heading))
    out_velocities[0] = wp.spatial_vector(
        wp.vec3(wp.cos(heading) * next_u - wp.sin(heading) * next_v,
                wp.sin(heading) * next_u + wp.cos(heading) * next_v, 0.0),
        wp.vec3(0.0, 0.0, next_yaw))
