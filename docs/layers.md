# Layered tracks (bridges)

A layered track stacks 2.5D height maps so a road can pass over itself. Run
from the repository root:

```bash
uv run warptracer demo bridge            # 120 s figure eight -> outputs/bridge.npz
uv run warptracer demo bridge --view     # then open the native Warp viewer
```

The car starts on the ground straight, drives **under** the bridge, around the
left loop, up the ramp, **over** the crossing, down into the right loop and
back. The JSON reports lap progress and wall-contact substeps.

## Model

Each layer is one height field plus a drivable mask on a shared node grid
(`heights`, `free`: `[layers, nx, ny]`). Layer 0 is the floor; higher layers are
decks. Every car carries an integer layer index.

- **Height and tilt.** Physics reads the car's own layer: bilinear height and a
  central-difference gradient give z, pitch and roll. Other layers are ignored,
  so a car under the bridge never sees the deck as ground.
- **Seams.** A car leaves its layer only where it drives off that layer's road
  onto another layer that is drivable *at the same height* (within 3 cm,
  `SEAM_TOLERANCE`). On the figure eight the seams are the far points of the two
  loops, where the ramps meet the floor. Everywhere else, leaving the road is a
  wall: the deck edge over the lower road is a wall for the car on the deck, and
  the embankment beside the lower road is a wall for the car below.
- **Collision.** Per-layer clearance fields, as for PNG maps, but tested against
  samples of the chassis outline (5 cm apart) rather than an enclosing circle,
  matching the oval's footprint check and the disparity controller's margins.
  Moves that reduce clearance at contact are rejected; moves that increase it
  are allowed, so a car touching a wall can steer away.
- **Sensing and replay.** The grids are turned into one triangle mesh: road
  surfaces, 0.3 m walls along road edges (open at seams), solid embankment under
  the ramps, and a deck slab with an underside over the lower road. LiDAR uses
  the mesh backend (a single height field cannot represent an overhang; grid
  LiDAR is rejected). For LiDAR only, the track sits inside a closed room so
  rays climbing off the deck still return, as they would indoors.

## Figure-eight geometry

`layers.figure_eight_bridge()` builds the example from parameters:

| Setting | Default |
| --- | --- |
| Loop radius (centerline) / loop centers | 3.0 m / x = ±6 m |
| Road width | 2.0 m |
| Bridge height | 0.7 m (about 2.3 ft), held for 3 m over the crossing |
| Ramp | Cosine profile, ~10 m each side, max grade about 13.6% |
| Crossing angle | 60° |
| Deck thickness / wall height | 0.08 m / 0.3 m |
| Seam overlap | Each layer's road runs 1 m past its seam |
| Grid | 0.05 m, about 23 × 10 m |

Recordings and the JSON sidecar store the generator parameters, not the grids.
Layered tracks built from arrays (for example a future scan importer) embed
compressed grids instead.

## Limits

- The lean model is planar: slopes tilt the chassis but gravity does not slow
  the car on the ramp, and it never leaves the ground.
- The disparity controller is unchanged (loop mode, as on the oval). Measured on
  the default geometry: about three laps in 120 s with zero wall contacts. Other
  geometry needs its own navigation check.
- PPO training does not support layered tracks yet.
- Warp 1.17's flat-shading path scrambles these meshes, so replay draws them
  with smooth shading (faces still look flat; quads share no vertices).
