# WarpTracer

A 3D sensor prototype for autonomous racing, built with NVIDIA Warp.

The standalone [RaceSense3D package](racesense3d/) provides triangle-mesh ray
casting, LiDAR and depth cameras, batched sensor poses, reusable device buffers,
geometry tests, and a self-contained Colab notebook. Vehicle physics is deferred.

[Open the sensor notebook in Google Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/sensor-prototype/racesense3d/notebooks/colab_sensors.ipynb)

## Run locally

```bash
cd racesense3d
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/sensors.py sensor_preview.png
```

See [the package README](racesense3d/README.md) for the sensor API and coordinate
conventions, and [validation notes](racesense3d/VALIDATION.md) for tested behavior
and remaining limitations. CPU validation: 21 passed, 1 skipped; CUDA and hosted
Colab execution remain unverified.

The existing root-level verification script and project files are retained.
