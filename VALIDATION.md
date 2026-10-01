# Validation

Results from 2026-10-01 UTC (2026-09-30 in California).

Environment: Linux x86-64, Python 3.12.14, Newton 1.6.0, Warp 1.17.0,
Viser 1.0.26. Dependencies were installed using the root `uv.lock`.

## Physics

- The four-second tilted-drop and wall-impact demos execute on CPU.
- The chassis settles at approximately 0.05991 m center height (0.06 m
  half-height); the roughly 0.09 mm resting penetration is within the documented
  test tolerance for iterative contacts.
- The sideways chassis approaches the outer barrier and stops on its inner side.
- Automated checks cover analytic free fall before impact, reset, mass, settling,
  quaternion normalization, floor penetration, and barrier containment.
- `uv run --locked --extra viz --extra dev pytest -q`: **4 passed, 1 skipped**.
- A separate headless CPU run completed four simulated seconds successfully.
- A separate CUDA smoke test runs when a driver is available. No NVIDIA driver
  was available here, so GPU behavior and performance remain unverified.

## Playback and notebook

- Both scenarios export standalone Viser HTML with scene geometry and timed poses.
- Exported recordings were decoded and checked for four seconds of timed geometry,
  position, and orientation messages. Each complete HTML file is about 2.86 MB.
- The browser draws recorded geometry; playback does not run Newton or require
  a persistent Python server.
- The notebook uses the preinstalled `uv`, clones `rigidbody-prototype`, and runs
  from the repository root. Its code structure is checked locally.
- Hosted Colab execution and browser playback have not been verified in this
  authoring environment. A Chromium download for visual QA failed.

## Scope

This is one free rigid body and a static track. There is no driving controller,
wheel contact model, suspension, sensor workload, RL wrapper, or environment
batching. Example dimensions, mass, and friction are starting values, not a
calibrated vehicle model. Tire/deck details are display geometry only.
