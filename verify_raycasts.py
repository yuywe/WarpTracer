import numpy as np
import warp as wp
import matplotlib.pyplot as plt

wp.init()

# ---------------------------------------------------------
# 1. SETUP GEOMETRY (10m x 10m x 3m Box)
# ---------------------------------------------------------
half_w, half_l, height = 5.0, 5.0, 3.0

vertices = np.array([
    [-half_w, -half_l, 0.0], [half_w, -half_l, 0.0],
    [half_w,  half_l, 0.0], [-half_w,  half_l, 0.0],
    [-half_w, -half_l, height], [half_w, -half_l, height],
    [half_w,  half_l, height], [-half_w,  half_l, height],
], dtype=np.float32)

faces = np.array([
    # Floor: normal must point UP (+Z) -> [0, 1, 2], [0, 2, 3]
    [0, 1, 2], [0, 2, 3],
    
    # Ceiling: normal must point DOWN (-Z) -> [4, 7, 6], [4, 6, 5]
    [4, 7, 6], [4, 6, 5],
    
    # Left wall (-X): normal must point RIGHT (+X)
    [0, 3, 7], [0, 7, 4],
    
    # Right wall (+X): normal must point LEFT (-X)
    [1, 5, 6], [1, 6, 2],
    
    # Back wall (-Y): normal must point FORWARD (+Y)
    [0, 4, 5], [0, 5, 1],
    
    # Front wall (+Y): normal must point BACK (-Y)
    [3, 2, 6], [3, 6, 7],
], dtype=np.int32)

mesh = wp.Mesh(
    points=wp.array(vertices, dtype=wp.vec3),
    indices=wp.array(faces.flatten(), dtype=int)
)

# ---------------------------------------------------------
# 2. KERNEL DEFINITION
# ---------------------------------------------------------
@wp.kernel
def cast_lidar(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    angles: wp.array(dtype=float),
    pitch_rad: float,
    ranges: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
    max_range: float
):
    tid = wp.tid()
    theta = angles[tid]
    
    # 3D ray direction accounting for pitch tilt
    cos_p = wp.cos(pitch_rad)
    direction = wp.vec3(
        wp.cos(theta) * cos_p,
        wp.sin(theta) * cos_p,
        -wp.sin(pitch_rad)
    )
    
    query = wp.mesh_query_ray(mesh_id, origin, direction, max_range)
    if query.result:
        ranges[tid] = query.t
        normals[tid] = query.normal
    else:
        ranges[tid] = max_range
        normals[tid] = wp.vec3(0.0, 0.0, 0.0)

# ---------------------------------------------------------
# 3. RUN & VERIFY TESTS
# ---------------------------------------------------------
num_beams = 1080
fov_rad = np.deg2rad(270.0)
angles_np = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, num_beams, dtype=np.float32)

wp_angles = wp.array(angles_np, dtype=float)
wp_ranges = wp.zeros(num_beams, dtype=float)
wp_normals = wp.zeros(num_beams, dtype=wp.vec3)
sensor_origin = wp.vec3(0.0, 0.0, 0.15)  # 15cm off floor

# --- Test A: Flat Level Raycast (Pitch = 0) ---
wp.launch(
    kernel=cast_lidar,
    dim=num_beams,
    inputs=[mesh.id, sensor_origin, wp_angles, 0.0, wp_ranges, wp_normals, 30.0]
)

ranges_flat = wp_ranges.numpy()
center_idx = num_beams // 2  # 0 deg heading (straight at +X wall at 5.0m)

assert np.isclose(ranges_flat[center_idx], 5.0, atol=1e-3), f"Expected 5.0m, got {ranges_flat[center_idx]}"
print("✓ Test A Passed: Planar forward hit = 5.000m")

# --- Test B: Chassis Pitch Down (5 deg) ---
pitch_5deg = float(np.deg2rad(5.0))
wp.launch(
    kernel=cast_lidar,
    dim=num_beams,
    inputs=[mesh.id, sensor_origin, wp_angles, pitch_5deg, wp_ranges, wp_normals, 30.0]
)

ranges_pitched = wp_ranges.numpy()
normals_pitched = wp_normals.numpy()

# Expected floor hit: 0.15 / sin(5°) ≈ 1.7207m
expected_floor_dist = 0.15 / np.sin(pitch_5deg)
assert np.isclose(ranges_pitched[center_idx], expected_floor_dist, atol=1e-2), "Floor hit mismatch"
assert np.isclose(normals_pitched[center_idx][2], 1.0, atol=1e-3), "Normal should point up (+Z)"
print(f"✓ Test B Passed: Floor clip distance = {ranges_pitched[center_idx]:.3f}m with normal {normals_pitched[center_idx]}")

# ---------------------------------------------------------
# 4. OPTIONAL VISUAL INSPECTION
# ---------------------------------------------------------
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(angles_np, ranges_flat, label='Level')
ax.plot(angles_np, ranges_pitched, label='5° Pitch Down')
ax.set_ylim(0, 8)
ax.legend()
plt.savefig("lidar_verification.png")
print("Saved polar verification plot to lidar_verification.png")