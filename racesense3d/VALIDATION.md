# Validation — 2026-09-30

- Original validation: Python 3.12.14, Linux x86_64; NVIDIA Warp 1.17.0;
  NumPy 2.3.5.
- `python -m pytest tests -q`: **21 passed, 1 skipped**.
- Includes 4,346 deterministic ray/pose comparisons against an independent
  scalar ray–axis-aligned-box slab calculation (41 origins × 106 directions).
- Other checks cover camera optical depth vs slant range, intrinsics/reprojection,
  batched poses, 3D rigid transforms, floor/elevation conventions, nearest hit,
  two-sided triangles, near-range occlusion, half-open range limits, validation
  errors, device input buffers and output-buffer reuse.
- The original embedded-source notebook's six code cells executed successfully
  in sequence in a shared Python namespace. The current notebook has three code
  cells and clones the GitHub branch instead; its cells pass syntax/structure
  checks. The sensor source and tests are unchanged.
- `uv sync --locked --extra dev` installed the package in a fresh `.venv`.
- `uv run --locked --extra dev pytest -q`: **21 passed, 1 skipped** with
  Python 3.12.14, Warp 1.17.0, NumPy 2.5.3, and Matplotlib 3.11.2.
- `uv run --locked --extra dev python examples/sensors.py` generated the preview.
- The current notebook's structure and three code cells pass syntax checks;
  setup uses Colab's Python via `UV_PYTHON`. Hosted Colab remains unverified.
- Both original plots generated successfully; the original preview was inspected.
- No NVIDIA driver was available. The CUDA parity test was skipped; CUDA kernels
  and performance have NOT been verified on a GPU in this environment.
- Full Jupyter execution was attempted but blocked by restricted socket binding
  (both TCP and IPC). The fallback above validates cell code, not Jupyter transport.
  Hosted Google Colab execution remains unverified.
- Original WarpORacer repository was read at
  e16e4473cb86e6ae2e1fa1b078e5a1e04b20d7f2. No source or assets were copied.
