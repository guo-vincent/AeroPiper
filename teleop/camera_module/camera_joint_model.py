"""
camera_joint_model.py

Bridges MediaPipe Pose + stereo depth to VRJointMapper.
Converts 3D body landmarks into the same 6D feature vector format
used by vr_joint_model.py so the RBF regressor works unchanged.

Feature layout (matches vr_joint_model.feature_from_pose):
  [0:3] dpos / pos_range   — wrist position relative to resting wrist
  [3:6] rotvec(dq) / pi    — forearm orientation relative to resting forearm

The "controller" analog here is the WRIST landmark.
The "orientation" analog is the forearm axis (elbow to wrist) expressed
as a quaternion rotation from the resting forearm direction.
"""

from __future__ import annotations

from typing import Optional, cast

import numpy as np

from vr_module.vr_joint_model import (
    Hand,
    VRJointMapper,
    HandReference,
    feature_from_pose,
    rotmat_to_quat_wxyz,
    load_calibration,
    build_dataset,
)

# ── Constants ──────────────────────────────────────────────────────────────────
POS_RANGE  = 0.6
ROT_RANGE  = float(np.pi)

# ── Geometry helpers ───────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-7 else v


def forearm_quat(
    elbow: np.ndarray,
    wrist: np.ndarray,
    index: Optional[np.ndarray] = None,
    pinky: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns a quaternion (w,x,y,z) representing the forearm's orientation.

    Builds a rotation matrix whose Z-axis points along elbow→wrist.
    The X-axis (right) is determined by the across-knuckle direction when
    hand landmarks are available, which correctly encodes forearm roll.

    Without hand landmarks the X-axis falls back to a world-up reference,
    which loses roll — forearm_supinated and forearm_pronated will then
    produce identical quaternions, poisoning those calibration poses.

    Args:
        elbow:  3D elbow landmark position
        wrist:  3D wrist landmark position
        index:  3D index-finger landmark (landmark 19/20 in PoseLandmarker)
        pinky:  3D pinky-finger landmark  (landmark 17/18 in PoseLandmarker)
    """
    fwd = _unit(wrist - elbow)  # Z axis — forearm direction

    if index is not None and pinky is not None:
        # pinky to index spans knuckles and physically rotates forearm
        # Project component along fwd so it stays perpendicular,
        # then use it as X (right) axis of the frame.
        hand_lateral = np.asarray(index, dtype=np.float64) - np.asarray(pinky, dtype=np.float64)
        hand_lateral = hand_lateral - np.dot(hand_lateral, fwd) * fwd
        n = float(np.linalg.norm(hand_lateral))
        if n > 1e-6:
            right = hand_lateral / n
            up    = _unit(np.cross(right, fwd))
            R = np.column_stack([right, up, fwd])
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
            return rotmat_to_quat_wxyz(R)
        # Fall through to world-up fallback if hand_lateral degenerate

    # Roll unobservable
    world_up = np.array([0.0, -1.0, 0.0])  # camera Y is down
    right = _unit(np.cross(fwd, world_up))
    up    = _unit(np.cross(right, fwd))
    R = np.column_stack([right, up, fwd])
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return rotmat_to_quat_wxyz(R)


def landmarks_to_feature(
    shoulder: np.ndarray,
    elbow:    np.ndarray,
    wrist:    np.ndarray,
    ref_wrist:        np.ndarray,
    ref_forearm_quat: np.ndarray,
    index:    Optional[np.ndarray] = None,
    pinky:    Optional[np.ndarray] = None,
    pos_range: float = POS_RANGE,
    rot_range: float = ROT_RANGE,
) -> np.ndarray:
    """
    Converts 3D arm landmarks to 6D feature compatible with VRJointMapper.

    Pass index and pinky finger landmarks so that forearm roll is captured.
    Without them the supinated/pronated calibration poses will be identical
    in feature space, making roll prediction impossible.
    """
    q_forearm = forearm_quat(elbow, wrist, index=index, pinky=pinky)
    return feature_from_pose(
        pos_m         = wrist,
        quat_wxyz     = q_forearm,
        ref_pos_m     = ref_wrist,
        ref_quat_wxyz = ref_forearm_quat,
        pos_range     = pos_range,
        rot_range     = rot_range,
    )

class ArmReference:
    """
    Captured once at startup (user holds 'resting' pose for ~2 s).
    Stores the resting wrist position and forearm quaternion (including roll).
    """
    def __init__(
        self,
        wrist: np.ndarray,
        elbow: np.ndarray,
        index: Optional[np.ndarray] = None,
        pinky: Optional[np.ndarray] = None,
    ):
        self.wrist = np.asarray(wrist, dtype=np.float64).copy()
        self.forearm_quat = forearm_quat(
            np.asarray(elbow, dtype=np.float64),
            np.asarray(wrist, dtype=np.float64),
            index=index,
            pinky=pinky,
        )

    def to_hand_reference(self) -> HandReference:
        """Cast to HandReference for VRJointMapper.predict_from_pose()."""
        return HandReference(
            pos_m     = self.wrist,
            quat_wxyz = self.forearm_quat,
        )


def _validate_side(side: str) -> Hand:
    """
    Validate that `side` is 'left' or 'right' and narrow its type to Hand.
    Raises ValueError at runtime for any unexpected value, and satisfies
    the type checker via cast().
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return cast(Hand, side)


# ── Runtime mapper ─────────────────────────────────────────────────────────────

class CameraArmMapper:
    """
    Drop-in runtime mapper for camera-based teleop.

    Usage:
        mapper = CameraArmMapper.from_calibration("camera_calibration.json")
        mapper.set_resting_reference("left",  wrist_3d, elbow_3d)
        mapper.set_resting_reference("right", wrist_3d, elbow_3d)
        joints_norm = mapper.predict("left", shoulder, elbow, wrist)
    """

    def __init__(
        self,
        vr_mapper: VRJointMapper,
        pos_range: float = POS_RANGE,
        rot_range: float = ROT_RANGE,
    ):
        self._mapper   = vr_mapper
        self._refs: dict[Hand, ArmReference] = {}
        self.pos_range = pos_range
        self.rot_range = rot_range

    @classmethod
    def from_calibration(cls, path: str) -> "CameraArmMapper":
        calib = load_calibration(path)
        Xl, Yl, Xr, Yr = build_dataset(calib)
        m = VRJointMapper(
            method    = calib.get("method", "rbf"),
            pos_range = calib.get("pos_range_m",  POS_RANGE),
            rot_range = calib.get("rot_range_rad", ROT_RANGE),
        )
        m.fit(Xl, Yl, Xr, Yr)
        return cls(
            m,
            pos_range = calib.get("pos_range_m",  POS_RANGE),
            rot_range = calib.get("rot_range_rad", ROT_RANGE),
        )

    def set_resting_reference(
        self,
        side:  str,
        wrist: np.ndarray,
        elbow: np.ndarray,
        index: Optional[np.ndarray] = None,
        pinky: Optional[np.ndarray] = None,
    ) -> None:
        """
        Call once when the user holds the 'resting' pose at runtime.
        Pass index and pinky landmarks so the resting roll is captured.
        """
        self._refs[_validate_side(side)] = ArmReference(wrist, elbow, index=index, pinky=pinky)

    def predict(
        self,
        side:     str,
        shoulder: np.ndarray,
        elbow:    np.ndarray,
        wrist:    np.ndarray,
        index:    Optional[np.ndarray] = None,
        pinky:    Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Returns 6 normalised joint targets in [-1, 1], or None if reference not set."""
        hand = _validate_side(side)
        if hand not in self._refs:
            return None
        ref = self._refs[hand]
        feat = landmarks_to_feature(
            shoulder, elbow, wrist,
            ref.wrist, ref.forearm_quat,
            index=index,
            pinky=pinky,
            pos_range=self.pos_range,
            rot_range=self.rot_range,
        )
        return self._mapper.predict_from_feature(hand, feat)