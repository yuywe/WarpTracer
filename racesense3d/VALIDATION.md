# Validation — 2026-09-30

- Python 3.12.14, Linux x86_64; NVIDIA Warp 1.17.0; NumPy 2.3.5.
- `python -m pytest tests -q`: **21 passed, 1 skipped**.
- Includes 4,346 deterministic ray/pose comparisons against an independent
  scalar ray–axis-aligned-box slab calculation (41 origins × 106 directions).
- Other checks cover camera optical depth vs slant range, intrinsics/reprojection,
  batched poses, 3D rigid transforms, floor/elevation conventions, nearest hit,
  two-sided triangles, near-range occlusion, half-open range limits, validation
  errors, device input buffers and output-buffer reuse.
- All six code cells in `notebooks/colab_sensors.ipynb` executed successfully in
  sequence in a shared Python namespace, including dependency setup, extracted
  package tests, visualization, batched camera observations and source export.
- Both plots generated successfully; the sensor preview was visually inspected.
- No NVIDIA driver was available. The CUDA parity test was skipped; CUDA kernels
  and performance have NOT been verified on a GPU in this environment.
- Full Jupyter execution was attempted but blocked by restricted socket binding
  (both TCP and IPC). The fallback above validates cell code, not Jupyter transport.
  Hosted Google Colab execution remains unverified.
- Original WarpORacer repository was read at
  e16e4473cb86e6ae2e1fa1b078e5a1e04b20d7f2. No source or assets were copied.
