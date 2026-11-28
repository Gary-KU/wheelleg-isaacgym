"""Flight detector and ramp geometry checks, without changing robot parameters."""

import sys
from pathlib import Path
import numpy as np, mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.flight import FlightCompliance
from sim2sim.build_model import build

f = FlightCompliance(2, 0.002, 17)
gravity = np.array([[0.0, 0.0, 9.81]] * 2)
f.reset([0, 1], gravity)
for _ in range(20):
    f.update(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.81]]))
assert f.air.tolist() == [True, False]
for _ in range(60):
    f.update(gravity)
assert not f.air.any() and f.blend.min() > 0.99
f.reset([0], gravity)
assert f.blend[0] == 1
m = mujoco.MjModel.from_xml_path(
    str(
        build(
            ROOT / "assets/wheelleg/mujoco/ramp_check.xml",
            terrain="ramp",
            flight_imu=True,
        )
    )
)
d = mujoco.MjData(m)
d.qpos[2] = 1
mujoco.mj_forward(m, d)
assert np.linalg.norm(d.sensor("flight_accel").data) < 3
for x, z in [(0.2, 0), (0.55, 0.015), (0.8, 0)]:
    hit = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        m, d, np.array([x, 0.8, 1.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, hit
    )
    assert abs(1 - distance - z) < 1e-6
print("flight hysteresis, landing blend, freefall IMU, ramp rays passed")
