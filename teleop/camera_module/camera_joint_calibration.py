"""
camera_joint_calibration.py

Calibration wizard for camera-based arm teleoperation.
Uses the MediaPipe Tasks API (v0.10.x+) — PoseLandmarker.

Writes:
  - camera_joint_calibration.json   (loadable by CameraArmMapper)
  - camera_joint_calibration_summary.txt

Usage (stereo):
    python teleop/camera_module/camera_joint_calibration.py --maps calibration/rectification/rect_maps.npz --rect calibration/rectification/rectification.yaml --left 0 --right 1

Usage (mono):
    python teleop/camera_module/camera_joint_calibration.py --mono --left 0

Model file (~29 MB) is auto-downloaded on first run to:
    teleop/camera_module/pose_landmarker_full.task
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from teleop.utils import setup_sys_path, get_monotonic_ts, ensure_model

setup_sys_path()

from teleop.vr_module import (
    feature_from_pose,
    quat_normalize,
    quat_average,
)
from camera_joint_model import (
    forearm_quat,
    POS_RANGE,
    ROT_RANGE,
)

# ── Model auto-download ────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent

# ── Landmark index constants ───────────────────────────────────────────────────
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16
_L_PINKY    = 17  # needed for forearm roll
_R_PINKY    = 18
_L_INDEX    = 19
_R_INDEX    = 20

# (shoulder, elbow, wrist, pinky, index)
_SIDE_INDICES: Dict[str, Tuple[int, int, int, int, int]] = {
    "left":  (_L_SHOULDER, _L_ELBOW, _L_WRIST, _L_PINKY, _L_INDEX),
    "right": (_R_SHOULDER, _R_ELBOW, _R_WRIST, _R_PINKY, _R_INDEX),
}

# Arm skeleton connections for OpenCV drawing
_ARM_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulder bar
]

# ── Calibration constants ──────────────────────────────────────────────────────
SAMPLE_SECS  = 3.0
POLL_HZ      = 30.0
BETWEEN_SECS = 8.0

OUT_DIR     = _MODULE_DIR
OUT_JSON    = OUT_DIR / "camera_joint_calibration.json"
OUT_SUMMARY = OUT_DIR / "camera_joint_calibration_summary.txt"


# ── Pose definitions ───────────────────────────────────────────────────────────

@dataclass
class PoseSpec:
    name: str
    description: str
    robot_targets_norm: Dict[str, List[float]]
    duration_s: float = SAMPLE_SECS


def _build_pose_list() -> List[PoseSpec]:
    return [
        PoseSpec(
            name="resting",
            description=(
                "Stand naturally, arms relaxed at your sides or in a comfortable\n"
                "neutral position. This is the REFERENCE pose — all other poses\n"
                "are measured relative to this one."
            ),
            robot_targets_norm={
                "left":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="arms_forward",
            description="Extend both arms straight forward, parallel to the floor.",
            robot_targets_norm={
                "left":  [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                "right": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="arms_up",
            description="Raise both arms straight up overhead.",
            robot_targets_norm={
                "left":  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "right": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="elbows_bent_90",
            description="Bend elbows 90 degrees, upper arms at sides, forearms pointing forward.",
            robot_targets_norm={
                "left":  [0.0, 0.2, 0.6, 0.0, 0.0, 0.0],
                "right": [0.0, 0.2, 0.6, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="arms_crossed",
            description="Cross arms over chest (right hand on left shoulder, vice versa).",
            robot_targets_norm={
                "left":  [-0.5, 0.3, 0.7, 0.0, 0.0, 0.0],
                "right": [0.5,  0.3, 0.7, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="arms_wide",
            description="T-pose: arms extended directly to the sides, parallel to floor.",
            robot_targets_norm={
                "left":  [-0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                "right": [0.8,  0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="forearm_supinated",
            description="Elbows at 90 degrees, palms facing UP (forearm supinated).",
            robot_targets_norm={
                "left":  [0.0, 0.2, 0.6, -0.8, 0.0, 0.0],
                "right": [0.0, 0.2, 0.6,  0.8, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="forearm_pronated",
            description="Elbows at 90 degrees, palms facing DOWN (forearm pronated).",
            robot_targets_norm={
                "left":  [0.0, 0.2, 0.6,  0.8, 0.0, 0.0],
                "right": [0.0, 0.2, 0.6, -0.8, 0.0, 0.0],
            },
        ),
    ]


# ── Drawing ────────────────────────────────────────────────────────────────────────

def _draw_pose(
    image: np.ndarray,
    norm_landmarks: list,
    connections: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
    thickness: int = 2,
) -> None:
    """Overlay pose skeleton onto image in-place using OpenCV."""
    h, w = image.shape[:2]
    pts = {
        i: (int(lm.x * w), int(lm.y * h))
        for i, lm in enumerate(norm_landmarks)
        if getattr(lm, "visibility", 1.0) > 0.3
    }
    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(image, pts[a], pts[b], color, thickness)
    for pt in pts.values():
        cv2.circle(image, pt, radius, color, -1)


# ── Landmark helpers ───────────────────────────────────────────────────────────

def _world_to_array(lm) -> np.ndarray:
    """Tasks world landmark (metres, hip-relative) → numpy (3,)."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _stereo_lift(
    norm_lm: Any,
    img_w: int,
    img_h: int,
    points_3d: np.ndarray,
    half_win: int = 3,
) -> Optional[np.ndarray]:
    """
    Lift a normalized landmark into 3D via stereo depth map.

    Samples a (2*half_win+1)² window around the landmark pixel and returns
    the per-channel median of valid points. This suppresses StereoSGBM
    speckle noise and holes that cause violent depth jumps at single pixels.
    """
    cx = int(np.clip(norm_lm.x * img_w, 0, img_w - 1))
    cy = int(np.clip(norm_lm.y * img_h, 0, img_h - 1))
    x0, x1 = max(0, cx - half_win), min(img_w, cx + half_win + 1)
    y0, y1 = max(0, cy - half_win), min(img_h, cy + half_win + 1)

    patch = points_3d[y0:y1, x0:x1].reshape(-1, 3)  # (N, 3)
    z     = patch[:, 2]
    valid = np.isfinite(z) & (z > 0.1) & (z < 5.0)
    if valid.sum() < 3:
        return None
    return np.median(patch[valid], axis=0)


def _best_3d(
    norm_lm,
    world_lm,
    img_w: int,
    img_h: int,
    points_3d: Optional[np.ndarray],
    min_vis: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Returns the best available 3D position for one landmark:
      1. Stereo reprojection (metric, camera frame) — preferred
      2. MediaPipe world landmark (metric, hip-relative) — fallback
      3. None if visibility is too low
    """
    if getattr(norm_lm, "visibility", 1.0) < min_vis:
        return None
    if points_3d is not None:
        stereo_pt = _stereo_lift(norm_lm, img_w, img_h, points_3d)
        if stereo_pt is not None:
            return stereo_pt
    return _world_to_array(world_lm)


# ── Core sampling loop ─────────────────────────────────────────────────────────

def sample_arm_landmarks(
    cap_left,
    cap_right,
    map1x, map1y, map2x, map2y, Q,
    stereo,
    landmarker: Any,  # mp_vision.PoseLandmarker
    duration_s: float,
    poll_hz: float,
    mono: bool = False,
) -> Dict:
    """
    Record arm landmark samples for `duration_s` seconds.
    Returns stats dict per side matching vr_joint_calibration's VR stats format
    so build_dataset() in vr_joint_model.py works unchanged.
    """
    dt = 1.0 / poll_hz
    samples: Dict[str, List] = {"left": [], "right": []}
    frames_total    = 0
    frames_detected = 0
    result          = None

    t_end = time.perf_counter() + duration_s

    while time.perf_counter() < t_end:
        t0 = time.perf_counter()

        if mono:
            ret, frame = cap_left.read()
            if not ret:
                continue
            rectL     = frame
            points_3d = None
        else:
            cap_left.grab()
            cap_right.grab()
            retL, frameL = cap_left.retrieve()
            retR, frameR = cap_right.retrieve()
            if not retL or not retR:
                continue
            rectL  = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
            rectR  = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)
            grayL  = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            grayR  = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
            disp   = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            points_3d = cv2.reprojectImageTo3D(disp, Q)

        img_h, img_w = rectL.shape[:2]

        # Tasks API needs an mp.Image in SRGB format + a timestamp in VIDEO mode
        rgb = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = landmarker.detect_for_video(mp_image, get_monotonic_ts())
        frames_total += 1

        display = rectL.copy()

        if result.pose_landmarks and result.pose_world_landmarks:
            frames_detected += 1
            norm_lms  = result.pose_landmarks[0]   # first detected person
            world_lms = result.pose_world_landmarks[0]

            _draw_pose(display, norm_lms, _ARM_CONNECTIONS)

            for side, (sh_i, el_i, wr_i, pk_i, idx_i) in _SIDE_INDICES.items():
                s  = _best_3d(norm_lms[sh_i],  world_lms[sh_i],  img_w, img_h, points_3d)
                e  = _best_3d(norm_lms[el_i],  world_lms[el_i],  img_w, img_h, points_3d)
                w  = _best_3d(norm_lms[wr_i],  world_lms[wr_i],  img_w, img_h, points_3d)
                pk = _best_3d(norm_lms[pk_i],  world_lms[pk_i],  img_w, img_h, points_3d)
                ix = _best_3d(norm_lms[idx_i], world_lms[idx_i], img_w, img_h, points_3d)
                if s is not None and e is not None and w is not None:
                    samples[side].append((w.copy(), forearm_quat(e, w, index=ix, pinky=pk)))

        remaining = max(0.0, t_end - time.perf_counter())
        cv2.putText(
            display, f"Recording... {remaining:.1f}s", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        cv2.imshow("Calibration", display)
        cv2.waitKey(1)

        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        
    print(f"  [DIAG] frames={frames_total}  pose_detected={frames_detected}"
          f"  ({100*frames_detected/max(frames_total,1):.0f}% detection rate)")
    if frames_detected == 0:
        print("  [DIAG] No pose detected at all — check:")
        print("         • Is your full upper body visible in the Calibration window?")
        print("         • Is the room well-lit with no strong backlight?")
        print("         • Try standing 1.5-2.5 m from the camera.")
    elif all(len(samples[s]) == 0 for s in ("left", "right")):
        # Pose detected but all landmarks below min_vis — print actual values
        print("  [DIAG] Pose detected but all arm landmarks below visibility threshold.")
        print("         Last frame landmark visibility (shoulder/elbow/wrist/pinky/index):")
        
        if result is not None and result.pose_landmarks:
            lms = result.pose_landmarks[0]
            for side, (sh_i, el_i, wr_i, pk_i, idx_i) in _SIDE_INDICES.items():
                vals = " / ".join(
                    f"{getattr(lms[i], 'visibility', -1.0):.2f}"
                    for i in (sh_i, el_i, wr_i, pk_i, idx_i)
                )
                print(f"         {side.upper()}: {vals}")

    stats: Dict = {}
    for side in ("left", "right"):
        s = samples[side]
        if len(s) < 5:
            stats[side] = {"ok": False, "n": len(s)}
            continue
        positions = np.array([x[0] for x in s], dtype=np.float64)
        quats     = np.array([x[1] for x in s], dtype=np.float64)
        stats[side] = {
            "ok":        True,
            "n":         len(s),
            "pos_mean":  positions.mean(axis=0).tolist(),
            "pos_std":   positions.std(axis=0).tolist(),
            "quat_mean": quat_average(quats).tolist(),
        }
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left",  type=int, default=0)
    ap.add_argument("--right", type=int, default=1)
    ap.add_argument("--maps",  default="", help="NPZ rect maps (stereo)")
    ap.add_argument("--rect",  default="", help="YAML with Q matrix (stereo)")
    ap.add_argument("--num_disparities", type=int, default=128)
    ap.add_argument("--blocksize",       type=int, default=5)
    ap.add_argument(
        "--mono", action="store_true",
        help="Single camera — uses world landmarks only (good for pre-stereo testing)",
    )
    args = ap.parse_args()

    model_path = ensure_model()

    cap_left  = cv2.VideoCapture(args.left, cv2.CAP_DSHOW)
    cap_right = map1x = map1y = map2x = map2y = Q = stereo = None

    if not args.mono:
        cap_right = cv2.VideoCapture(args.right, cv2.CAP_DSHOW)
        maps  = np.load(args.maps)
        map1x, map1y = maps["map1x"], maps["map1y"]
        map2x, map2y = maps["map2x"], maps["map2y"]
        fs = cv2.FileStorage(args.rect, cv2.FILE_STORAGE_READ)
        Q  = fs.getNode("Q").mat()
        fs.release()
        nd = args.num_disparities
        if nd % 16:
            nd = (nd // 16 + 1) * 16
        bs = args.blocksize
        stereo = cv2.StereoSGBM.create(
            minDisparity=0, numDisparities=nd, blockSize=bs,
            P1=8 * 3 * bs**2, P2=32 * 3 * bs**2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=50, speckleRange=2,
        )
        print("[INFO] Stereo mode — reprojected depth with world landmark fallback.")
    else:
        print("[INFO] Mono mode — MediaPipe world landmarks only (hip-relative metres).")

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    lm_options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    poses       = _build_pose_list()
    calibration: Dict = {
        "version":       2,
        "created":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode":          "mono" if args.mono else "stereo",
        "pos_range_m":   POS_RANGE,
        "rot_range_rad": ROT_RANGE,
        "method":        "rbf",
        "poses":         [],
    }

    with mp_vision.PoseLandmarker.create_from_options(lm_options) as landmarker:
        print(f"\n[INFO] {len(poses)} poses to capture. Each: {SAMPLE_SECS}s hold + {BETWEEN_SECS}s transition.\n")
        input("Press ENTER when you are in front of the camera and ready...")

        for i, pose in enumerate(poses):
            print(f"\n{'─' * 60}")
            print(f"POSE {i + 1}/{len(poses)}: {pose.name.upper()}")
            print(f"{'─' * 60}")
            print(pose.description)

            if i > 0:
                print(f"\nMove into position. Recording in {int(BETWEEN_SECS)}s...")
                t_end = time.perf_counter() + BETWEEN_SECS
                while time.perf_counter() < t_end:
                    ret, frame = cap_left.read()
                    if ret:
                        cv2.putText(
                            frame, f"Next: {pose.name}  {t_end - time.perf_counter():.1f}s",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2,
                        )
                        cv2.imshow("Calibration", frame)
                        cv2.waitKey(1)
            else:
                input("Press ENTER to start recording the RESTING pose...")

            print("  RECORDING NOW — hold steady...")
            vr_stats = sample_arm_landmarks(
                cap_left, cap_right,
                map1x, map1y, map2x, map2y, Q,
                stereo, landmarker,
                SAMPLE_SECS, POLL_HZ,
                mono=args.mono,
            )

            calibration["poses"].append({
                "name":               pose.name,
                "description":        pose.description,
                "duration_s":         pose.duration_s,
                "robot_targets_norm": pose.robot_targets_norm,
                "vr":                 vr_stats,  # key kept as "vr" for build_dataset() compat
            })

            for side in ("left", "right"):
                s = vr_stats.get(side, {})
                if s.get("ok"):
                    print(f"  {side.upper()}: n={s['n']}  pos_std_max={np.max(np.array(s['pos_std'])):.4f} m")
                else:
                    print(f"  {side.upper()}: NO DATA (n={s.get('n', 0)}) — check visibility")

    # ── Compute features relative to resting ──────────────────────────
    poses_out = calibration["poses"]
    ref = next((p for p in poses_out if p["name"] == "resting"), None)
    if ref is None or not all(ref["vr"].get(h, {}).get("ok") for h in ("left", "right")):
        print("\n[ERROR] Resting pose missing or incomplete — re-run calibration.")
        cap_left.release()
        if cap_right is not None:
            cap_right.release()
        cv2.destroyAllWindows()
        return

    ref_pos  = {h: np.array(ref["vr"][h]["pos_mean"])                  for h in ("left", "right")}
    ref_quat = {h: quat_normalize(np.array(ref["vr"][h]["quat_mean"])) for h in ("left", "right")}

    for p in poses_out:
        feats: Dict = {}
        for h in ("left", "right"):
            v = p.get("vr", {}).get(h, {})
            if not v.get("ok"):
                continue
            feats[h] = feature_from_pose(
                np.array(v["pos_mean"]),
                quat_normalize(np.array(v["quat_mean"])),
                ref_pos[h], ref_quat[h],
                POS_RANGE, ROT_RANGE,
            ).tolist()
        p["features"] = feats

    # ── Save ──────────────────────────────────────────────────────────
    OUT_JSON.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    lines = [
        "CAMERA ARM CALIBRATION SUMMARY",
        f"Created: {calibration['created']}",
        f"Mode:    {calibration['mode']}",
        "",
        "Joint order: [j1_base, j2_shoulder, j3_elbow, j4_forearm_roll, j5_wrist, j6_wrist_roll]",
    ]
    for p in poses_out:
        lines.append(f"\nPOSE: {p['name']}")
        for h in ("left", "right"):
            tgt  = p["robot_targets_norm"].get(h)
            feat = p.get("features", {}).get(h)
            lines.append(f"  {h.upper()}: target={tgt}  feature={feat}")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{'=' * 60}\nCALIBRATION SAVED\n  JSON   : {OUT_JSON}\n  Summary: {OUT_SUMMARY}")
    print("\nNext:\n  python teleop/camera_module/arm_pose_teleop.py "
          "--cal teleop/camera_module/camera_joint_calibration.json")

    cap_left.release()
    if cap_right is not None:
        cap_right.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()