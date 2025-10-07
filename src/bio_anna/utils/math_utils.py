from __future__ import annotations

from scipy.spatial.transform import Rotation as R


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return (x, y, z, w) quaternion corresponding to a planar yaw angle."""
    q = R.from_euler('z', yaw).as_quat()
    return q[0], q[1], q[2], q[3]


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw angle extracted from a quaternion."""
    return R.from_quat([x, y, z, w]).as_euler('xyz')[2]
