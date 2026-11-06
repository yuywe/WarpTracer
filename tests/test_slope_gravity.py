"""Slope gravity in the shared bicycle step, on analytic planes and the real tracks."""
import numpy as np
import pytest
import warp as wp

from warptracer.driving import DriveConfig
from warptracer.execution import TransitionRunner
from warptracer.layer_kernels import integrate_layers, surface_rotation
from warptracer.layers import figure_eight_bridge
from warptracer.lean import bicycle_params
from warptracer.scene import Track, Vehicle
from warptracer.simulation import Simulation
from warptracer.terrain import OvalTrack

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]
G = 9.81
DT = 1 / 240
RIDE = .1


class Plane:
    """One car on a uniform z = gx*x + gy*y layer, stepped by the real layer kernel."""

    def __init__(self, gx=0., gy=0., heading=0., speed=0., command=(0., 0., 0., -1.), device="cpu",
                 drag=0., max_braking=4., n=401, cell=.25):
        self.device, self.cell = device, cell
        self.params = bicycle_params(Vehicle(), DriveConfig(), DT)
        self.params.drag, self.params.max_braking = drag, max_braking
        self.origin = -.5 * n * cell
        centers = self.origin + (np.arange(n) + .5) * cell
        x, y = np.meshgrid(centers, centers, indexing="ij")
        make = lambda values, dtype: wp.array(values[None], dtype=dtype, device=device)
        self.heights = make((gx * x + gy * y).astype(np.float32), float)
        self.free = make(np.ones((n, n), np.int32), int)
        self.clearance = make(np.full((n, n), 1e3, np.float32), float)
        rotation = surface_rotation(heading, gx, gy)
        self.poses = wp.array([wp.transform(wp.vec3(0., 0., 0.), rotation)], dtype=wp.transform, device=device)
        self.motion = wp.array([wp.vec4(speed, 0., 0., heading)], dtype=wp.vec4, device=device)
        self.commands = wp.array([wp.vec4(*command)], dtype=wp.vec4, device=device)
        self.controls = wp.zeros(1, dtype=wp.vec3, device=device)
        self.collision = wp.zeros(1, dtype=int, device=device)
        self.velocities = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self.contacts = wp.zeros(1, dtype=int, device=device)
        self.layers = wp.zeros(1, dtype=int, device=device)

    def step(self, substeps=1, count=1):
        for _ in range(count):
            wp.launch(integrate_layers, dim=1, inputs=[self.poses, self.motion, self.commands, self.controls,
                self.collision, self.velocities, self.contacts, self.layers, self.heights, self.free,
                self.clearance, wp.vec2(self.origin, self.origin), self.cell, RIDE, .1, self.params, substeps],
                device=self.device)
        wp.synchronize_device(self.device)

    @property
    def state(self):
        return self.motion.numpy()[0], self.poses.numpy()[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grade,sign", [(.1, -1), (-.1, 1)])
def test_coasting_matches_gravity_along_a_plane(device, grade, sign):
    car = Plane(gx=grade, speed=2., device=device)
    car.step(count=240)
    motion, pose = car.state
    expected = 2. + sign * G * abs(grade) / np.hypot(1, grade)
    assert motion[0] == pytest.approx(expected, abs=1e-3)
    assert pose[2] - RIDE == pytest.approx(grade * pose[0], abs=1e-2)


def test_released_car_rolls_backward_down_a_ramp():
    car = Plane(gx=.1)
    car.step(count=120)
    motion, pose = car.state
    assert motion[0] < -.4 and pose[0] < -.1 and pose[2] < RIDE


def test_round_trip_over_a_ramp_returns_the_entry_speed():
    car = Plane(gx=.136, speed=2.)
    car.step(count=120)
    peak = car.state[1][2]
    assert peak > .1
    while car.state[1][0] > 0:
        car.step(count=1)
    assert abs(car.state[0][0]) == pytest.approx(2., rel=.01)


@pytest.mark.parametrize("device", DEVICES)
def test_brakes_hold_a_stopped_car_on_a_ramp(device):
    car = Plane(gx=.136, command=(0., 1., 0., -1.), device=device)
    car.step(count=1200)
    motion, pose = car.state
    assert abs(motion[0]) < 1e-5 and abs(pose[0]) < 1e-3


def test_brakes_stop_a_car_climbing_or_descending_without_later_creep():
    for grade in (.136, -.136):
        car = Plane(gx=grade, speed=1.5, command=(0., 1., 0., -1.))
        car.step(count=480)
        first = car.state[1][0]
        car.step(count=480)
        assert abs(car.state[0][0]) < 1e-3 and abs(car.state[1][0] - first) < 1e-3


def test_weak_brakes_cannot_hold_a_steep_slope():
    car = Plane(gx=.2, command=(0., 1., 0., -1.), max_braking=.5)
    car.step(count=480)
    assert car.state[1][0] < -.1


def test_cross_slope_pushes_the_car_sideways_downhill():
    # Driving along +x with the ground falling toward -y.
    car = Plane(gy=.1, speed=1.)
    car.step(count=240)
    motion, pose = car.state
    assert motion[1] < -1e-3 and pose[1] < 0 and pose[2] < RIDE
    # Driving across the slope, gravity has no component along the chassis.
    assert motion[0] == pytest.approx(1., abs=.05)


def test_throttle_is_weaker_uphill_and_stronger_downhill():
    command = (1., 0., 0., -1.)
    speeds = {grade: Plane(gx=grade, command=command) for grade in (.1, 0., -.1)}
    for car in speeds.values():
        car.step(count=120)
    up, flat, down = (speeds[grade].state[0][0] for grade in (.1, 0., -.1))
    # 0.5 s of 2 m/s^2 with and without gravity's contribution.
    assert flat == pytest.approx(1., abs=1e-3)
    assert up == pytest.approx(1. - .5 * G * .1 / np.hypot(1, .1), abs=1e-3)
    assert down == pytest.approx(1. + .5 * G * .1 / np.hypot(1, .1), abs=1e-3)


def test_flat_ground_has_exactly_no_slope_force():
    drag = .15
    sim = Simulation(track=Track(length=40, width=10), scenario="drive", device="cpu",
                     drive=DriveConfig(rolling_drag=drag))
    motion = sim.lean_motion.numpy()
    motion[0, 0] = 2.
    sim.lean_motion.assign(motion)
    for _ in range(240):
        sim.step()
    expected = 2. * (1 - DT * 4 * drag / sim.vehicle.mass) ** 240
    assert sim.lean_motion.numpy()[0, 0] == pytest.approx(expected, rel=1e-4)
    assert sim.lean_motion.numpy()[0, 1] == 0
    # Same reading from a perfectly flat oval: the road code adds no gravity either.
    flat = OvalTrack(elevation=0.)
    sim = Simulation(track=flat, scenario="drive", device="cpu",
                     drive=DriveConfig(rolling_drag=drag))
    motion = sim.lean_motion.numpy()
    motion[0, 0] = 2.
    sim.lean_motion.assign(motion)
    for _ in range(24):
        sim.step()
    assert sim.lean_motion.numpy()[0, 0] == pytest.approx(2. * (1 - DT * 4 * drag / sim.vehicle.mass) ** 24, rel=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_oval_coasting_trades_speed_for_height(device):
    # The spawn at x=0 heads +x, straight up the oval's steepest grade.
    sim = Simulation(track=OvalTrack(), scenario="drive", device=device,
                     drive=DriveConfig(rolling_drag=1e-6))
    motion = sim.lean_motion.numpy()
    motion[0, 0] = 1.5
    sim.lean_motion.assign(motion)
    start = sim.state.body_q.numpy()[0, 2]
    for _ in range(120):
        sim.step()
    wp.synchronize_device(sim.device)
    u = sim.lean_motion.numpy()[0, 0]
    climbed = sim.state.body_q.numpy()[0, 2] - start
    assert climbed > .03 and u < 1.5
    assert .5 * u * u + G * climbed == pytest.approx(.5 * 1.5 ** 2, rel=.01)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_substeps_match_single_steps_on_a_slope(device):
    command = (.5, 0., .1, -1.)
    single, fused = (Plane(gx=.1, gy=.05, speed=1., command=command, device=device) for _ in range(2))
    single.step(count=64)
    fused.step(substeps=4, count=16)
    np.testing.assert_array_equal(single.state[0], fused.state[0])
    np.testing.assert_array_equal(single.state[1], fused.state[1])


@pytest.mark.parametrize("track", [OvalTrack(), figure_eight_bridge()], ids=["oval", "bridge"])
@pytest.mark.parametrize("device", DEVICES)
def test_slope_gravity_keeps_graph_and_fused_parity(device, track):
    def runner(backend, integrator):
        sim = Simulation(track=track, scenario="drive", device=device, num_envs=4)
        sim.set_action(throttle=.6, steering=.05)
        return TransitionRunner(sim, backend=backend, substeps=4, integrator=integrator)

    reference = runner("eager", "unfused")
    for backend, integrator in (("eager", "fused"), ("graph", "unfused"), ("graph", "fused")):
        candidate = runner(backend, integrator)
        reference.reset()
        for sim in (reference.sim, candidate.sim):
            sim.set_action(throttle=.6, steering=.05)
        for _ in range(60):
            reference.advance()
            candidate.advance()
        wp.synchronize_device(reference.sim.device)
        np.testing.assert_allclose(candidate.sim.lean_motion.numpy(), reference.sim.lean_motion.numpy(), atol=1e-4)
        np.testing.assert_allclose(candidate.sim.state.body_q.numpy(), reference.sim.state.body_q.numpy(), atol=1e-4)


def test_bridge_ramp_slows_a_coasting_car():
    # Start on the route's steepest climb, coasting; gravity alone must slow it.
    track = figure_eight_bridge()
    sim = Simulation(track=track, scenario="drive", device="cpu")
    path = track.path(sim.drive.ride_height(sim.vehicle))
    qx, qy, qz, qw = path[:, 3:].T
    climbing = int(np.argmax(2 * (qx * qz - qy * qw)))
    assert track.route_layers[climbing] == 1
    sim.state.body_q.assign(path[climbing:climbing + 1])
    sim.layers.assign(np.array([1], np.int32))
    heading = np.arctan2(2 * (qw[climbing] * qz[climbing] + qx[climbing] * qy[climbing]),
                         1 - 2 * (qy[climbing] ** 2 + qz[climbing] ** 2))
    sim.lean_motion.assign(np.array([[1.5, 0., 0., heading]], np.float32))
    start = sim.state.body_q.numpy()[0, 2]
    for _ in range(120):
        sim.step()
    assert sim.state.body_q.numpy()[0, 2] > start
    assert sim.lean_motion.numpy()[0, 0] < 1.5 - .4
