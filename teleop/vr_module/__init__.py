"""
VR calibration + mapping modules (VR -> AeroPiper joint targets).

This directory contains the VR calibration + mapping code.
"""

# teleop/vr_module/__init__.py

from .vr_joint_calibration import (
    find_controllers,
    openvr_pose_to_pos_quat,
    compute_mean_pose,
    record_pose_samples,
    default_poses,
    recommended_pose_templates,
    extra_user_pose_templates,
    generate_inbetween_between_core,
    generate_coverage_targets,
    create_preview_robot,
    countdown_wait,
    main as run_calibration_wizard
)

from .vr_joint_model import (
    clamp01,
    quat_normalize,
    quat_mul,
    quat_inv,
    quat_to_rotvec,
    rotmat_to_quat_wxyz,
    quat_average,
    feature_from_pose,
    pairwise_sq_dists,
    load_calibration,
    build_dataset,
    RBFRegressor,
    NearestRegressor,
    LinearRegressor,
    VRJointMapper,
    HandReference
)

__all__ = [
    "find_controllers",
    "openvr_pose_to_pos_quat",
    "compute_mean_pose",
    "record_pose_samples",
    "create_preview_robot",
    "run_calibration_wizard",
    "quat_normalize",
    "feature_from_pose",
    "load_calibration",
    "build_dataset",
    "VRJointMapper",
    "RBFRegressor",
    "HandReference",
]


