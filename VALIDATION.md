# Validation

Checked on 2026-10-01 with Linux x86-64, Python 3.12.14, Newton 1.6.0,
Warp 1.17.0, and Viser 1.0.26, using the existing root `uv.lock`.

## Automated behavior checks

`uv run --locked --extra viz --extra dev pytest -q`: **12 passed, 2 skipped**.

Driving checks cover support at rest, reset of body and controls, forward
acceleration, braking to rest without appreciable reverse motion, coasting
versus braking, symmetric left/right turns, steering rate limits, zero-grip
propulsion, airborne behavior, invalid controls, and a wall collision under
continued throttle. Existing free-fall, floor-contact, barrier-impact, and
configuration checks also pass.

CUDA drop and driving checks are included but skipped here because no NVIDIA
driver is available. GPU behavior and performance remain unverified.

## Ten-second driving demonstrations

| Scenario | Measured CPU result |
| --- | --- |
| Accelerate/brake | Peak speed 1.688 m/s; stopped by the end; 2.9 m total forward travel |
| Circle | Approximately 0.938 m/s steady speed; 6.65 radians of heading change, completing a full lap |
| S-turn | Peak speed 0.849 m/s; yaw rate changes sign; stopped by the end |

All three trajectories remain inside their walls and export valid ten-second
Viser recordings. Each recording contains exactly six boxes and no added meshes.
The NPZ files contain 301 aligned pose, velocity, and control samples at 30 Hz.
A five-second headless accelerate/brake run also completes successfully.

Additional checks:

- Driving into a wall under continued speed demand stops the box at the wall
  face (center x approximately 1.740 m in a four-meter enclosure).
- A two-second constant-steering run at 120, 240, and 480 Hz produces final XY
  positions differing by less than 9 mm between the lowest and highest rate.
  This checks the default parameters, not arbitrary spring stiffnesses.
- The notebook's code cells compile. Its exact Git setup sequence succeeds
  twice in succession from both an older single-branch rigid-body clone and a
  fresh vehicle-controls clone, using temporary local clones for this check.
- Actual hosted Colab execution and rendered browser playback have not been
  checked in this environment. Exported scene data was decoded and inspected.

## Model scope

There is one six-degree-of-freedom rigid box. Its four unrendered contact points
use spring/damper support and friction-limited longitudinal/lateral forces.
Only the flat z=0 floor supports the tires. Steering rotates both front contact
directions equally. Propulsion is distributed across all four points. Braking
is smoothed below 0.2 m/s to reduce suspension recoil near rest.

This is a simple force model with illustrative parameters. It does not model
wheel spin, calibrated tire slip curves, Ackermann geometry, reverse throttle,
or tire support on uneven terrain. The speed helper is proportional and can
have steady error under drag. Sensors, disparity extender, parallel environments,
and PPO remain subsequent milestones.
