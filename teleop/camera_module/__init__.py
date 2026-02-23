"""
Camera calibration + mapping modules (camera -> AeroPiper joint targets).

This directory contains the camera calibration + mapping code.
"""

from .camera_joint_model import CameraArmMapper
from .arm_pose_teleop import run

from .arm_pose_teleop import (
    _L_SHOULDER, _R_SHOULDER,
    _L_ELBOW, _R_ELBOW,
    _L_WRIST, _R_WRIST,
    _SIDE_INDICES,
    _ARM_CONNECTIONS
)

__all__ = [
    "CameraArmMapper",
    "run",
    "_L_SHOULDER", "_R_SHOULDER",
    "_L_ELBOW", "_R_ELBOW",
    "_L_WRIST", "_R_WRIST",
    "_SIDE_INDICES",
    "_ARM_CONNECTIONS"
]