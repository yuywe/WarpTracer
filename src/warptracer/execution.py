"""Fixed-size transitions, executed eagerly or replayed from a Warp graph."""
import warp as wp


@wp.kernel
def circle_command(commands: wp.array(dtype=wp.vec4), clock: wp.array(dtype=int), dt: float):
    speed = float(0.0)
    if float(clock[0]) * dt >= 0.5:
        speed = 1.0
    commands[wp.tid()] = wp.vec4(0.0, 0.0, 0.25, speed)


@wp.kernel
def advance_clock(clock: wp.array(dtype=int), substeps: int):
    clock[0] += substeps


class TransitionRunner:
    """Own the simulation's stepping while active; reset through this runner.

    A transition has an even number of physics substeps and, when enabled, one
    LiDAR scan at the end. Even counts restore the state-buffer parity on every
    graph replay. The graph reads commands from stable device buffers.
    """
    def __init__(self, sim, backend="eager", substeps=4, controller="manual", integrator="unfused"):
        if backend not in ("eager", "graph"):
            raise ValueError("backend must be eager or graph")
        if not isinstance(substeps, int) or substeps < 2 or substeps % 2:
            raise ValueError("substeps must be a positive even integer >= 2")
        if sim.physics_hz % substeps:
            raise ValueError("substeps must divide physics_hz")
        if sim.lidar is not None and sim.lidar_stride != substeps:
            raise ValueError("Transition runner requires one LiDAR scan per transition")
        if controller not in ("manual", "circle"):
            raise ValueError("controller must be manual or circle")
        if controller == "circle" and (not sim.driving or sim.drive.max_steering < .25):
            raise ValueError("Circle benchmark requires a driving car with >= .25 rad steering")
        if integrator not in ("unfused", "fused"):
            raise ValueError("integrator must be unfused or fused")
        if integrator == "fused" and sim.engine != "lean":
            raise ValueError("Fused integration requires lean physics")
        self.integrator = integrator
        self.sim, self.backend, self.substeps, self.controller = sim, backend, substeps, controller
        self.clock = wp.zeros(1, dtype=int, device=sim.model.device)
        self.graph = None
        self.reset()
        if backend == "graph":
            # Materialize lazy solver allocations and load kernels before capture.
            self._operations()
            wp.synchronize_device(sim.model.device)
            self.reset()
            with wp.ScopedCapture(device=sim.model.device, force_module_load=False) as capture:
                self._operations()
            self.graph = capture.graph
            self.reset()

    @property
    def graph_kind(self):
        if self.backend == "eager":
            return None
        return "cuda" if self.sim.model.device.is_cuda else "cpu-api-capture"

    def _operations(self):
        sim = self.sim
        if self.controller == "circle":
            wp.launch(circle_command, dim=sim.num_envs, inputs=[sim.device_commands, self.clock, sim.dt],
                      device=sim.model.device)
        if self.integrator == "fused":
            sim._fused_physics_steps(self.substeps)
        else:
            for _ in range(self.substeps):
                sim._physics_step()
        if sim.lidar is not None:
            sim.lidar.update(sim.state, sim.body, 0.0)
        wp.launch(advance_clock, dim=1, inputs=[self.clock, self.substeps], device=sim.model.device)

    def reset(self):
        self.sim.reset()
        self.clock.zero_()

    def advance(self):
        if self.graph is None:
            self._operations()
        else:
            wp.capture_launch(self.graph)
        self.sim.steps += self.substeps
        if self.controller == "circle":
            # A device controller has changed the buffer independently of setters.
            self.sim._uploaded_command = None
        if self.sim.lidar is not None:
            self.sim.lidar.timestamp = self.sim.steps * self.sim.dt
