"""
arm_pose_teleop.py

Real-time arm teleoperation using MediaPipe Tasks PoseLandmarker + stereo depth.
Feeds 3D wrist/elbow/shoulder positions into CameraArmMapper (RBF regressor)
to produce 6 normalized joint targets for the Piper / AeroPiper arm.

Run as a background thread alongside gui.py.

Model file (~29 MB) is auto-downloaded on first run to the same directory as this script.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as _mp_python 
from mediapipe.tasks.python import vision as _mp_vision

# ── Path setup ─────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_TELEOP_DIR = _MODULE_DIR.parent
_REPO_ROOT  = _TELEOP_DIR.parent

for p in (_REPO_ROOT, _TELEOP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from camera_module.camera_joint_model import (
    CameraArmMapper
)

# ── Model auto-download ────────────────────────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_PATH = _MODULE_DIR / "pose_landmarker_full.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        print(f"[INFO] Downloading pose landmarker model (~29 MB) to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Download complete.")
    return _MODEL_PATH

# __ TIMER ______________________________________________________________________

_LAST_TS_MS = -1

def get_monotonic_ts() -> int:
    global _LAST_TS_MS
    current_ts = int(time.perf_counter() * 1000)
    if current_ts <= _LAST_TS_MS:
        current_ts = _LAST_TS_MS + 1
    _LAST_TS_MS = current_ts
    return current_ts

# ── Landmark indices ───────────────────────────────────────────────────────────
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16
_L_PINKY    = 17
_R_PINKY    = 18
_L_INDEX    = 19
_R_INDEX    = 20

# (shoulder, elbow, wrist, pinky, index)
_SIDE_INDICES: Dict[str, tuple] = {
    "left":  (_L_SHOULDER, _L_ELBOW, _L_WRIST, _L_PINKY, _L_INDEX),
    "right": (_R_SHOULDER, _R_ELBOW, _R_WRIST, _R_PINKY, _R_INDEX),
}

_ARM_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
]

# ── Exponential moving average filter ─────────────────────────────────────────

class EMAFilter:
    def __init__(self, alpha: float = 0.3, n: int = 6):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None
        self.n = n

    def update(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.copy()
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


# ── Drawing ────────────────────────────────────────────────────────────────────

def _draw_pose(image: np.ndarray, norm_lms: list, color: tuple = (0, 255, 0)) -> None:
    h, w = image.shape[:2]
    pts = {
        i: (int(lm.x * w), int(lm.y * h))
        for i, lm in enumerate(norm_lms)
        if getattr(lm, "visibility", 1.0) > 0.3
    }
    for a, b in _ARM_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(image, pts[a], pts[b], color, 2)
    for pt in pts.values():
        cv2.circle(image, pt, 5, color, -1)


# ── 3D landmark helpers ────────────────────────────────────────────────────────

def _world_to_array(lm: Any) -> np.ndarray:
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

    Samples a (2*half_win+1)² window and returns the per-channel median of
    valid points. Suppresses StereoSGBM speckle/holes that cause joint snapping
    when a single noisy pixel is sampled.
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
    norm_lm: Any,
    world_lm: Any,
    img_w: int,
    img_h: int,
    points_3d: Optional[np.ndarray],
    min_vis: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Best available 3D position for one landmark:
      1. Stereo reprojection (metric, camera frame) — preferred
      2. MediaPipe world landmark (metric, hip-relative) — fallback
      3. None if visibility < min_vis
    """
    if getattr(norm_lm, "visibility", 1.0) < min_vis:
        return None
    if points_3d is not None:
        pt = _stereo_lift(norm_lm, img_w, img_h, points_3d)
        if pt is not None:
            return pt
    return _world_to_array(world_lm)


# ── Resting-pose capture ───────────────────────────────────────────────────────

def capture_resting_reference(
    cap_left: Any,
    cap_right: Any,
    map1x: Any, map1y: Any, map2x: Any, map2y: Any, Q: Any,
    stereo: Any,
    landmarker: Any,
    mapper: CameraArmMapper,
    duration_s: float = 2.0,
    poll_hz: float = 30.0,
    mono: bool = False,
) -> bool:
    """
    Ask the user to hold the resting pose for `duration_s` seconds and
    set the reference on the mapper. Returns True on success.
    """
    print(f"\n[TELEOP] Hold RESTING pose for {duration_s:.0f}s...")
    samples: Dict[str, list] = {"left": [], "right": []}
    dt = 1.0 / poll_hz
    t_end = time.perf_counter() + duration_s

    while time.perf_counter() < t_end:
        t0 = time.perf_counter()

        if mono:
            ret, frame = cap_left.read()
            if not ret:
                continue
            rectL: np.ndarray = frame
            points_3d: Optional[np.ndarray] = None
        else:
            cap_left.grab()
            cap_right.grab()
            retL, frameL = cap_left.retrieve()
            retR, frameR = cap_right.retrieve()
            if not retL or not retR:
                continue
            rectL     = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
            rectR     = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)
            grayL     = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            grayR     = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
            disp      = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            points_3d = cv2.reprojectImageTo3D(disp, Q)

        img_h, img_w = rectL.shape[:2]
        rgb      = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, get_monotonic_ts())

        display = rectL.copy()
        if result.pose_landmarks and result.pose_world_landmarks:
            norm_lms  = result.pose_landmarks[0]
            world_lms = result.pose_world_landmarks[0]
            _draw_pose(display, norm_lms, color=(255, 200, 0))
            for side, (sh_i, el_i, wr_i, pk_i, idx_i) in _SIDE_INDICES.items():
                e  = _best_3d(norm_lms[el_i],  world_lms[el_i],  img_w, img_h, points_3d)
                w  = _best_3d(norm_lms[wr_i],  world_lms[wr_i],  img_w, img_h, points_3d)
                pk = _best_3d(norm_lms[pk_i],  world_lms[pk_i],  img_w, img_h, points_3d)
                ix = _best_3d(norm_lms[idx_i], world_lms[idx_i], img_w, img_h, points_3d)
                if e is not None and w is not None:
                    samples[side].append((e.copy(), w.copy(), pk, ix))

        remaining = max(0.0, t_end - time.perf_counter())
        cv2.putText(display, f"RESTING reference... {remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
        cv2.imshow("Teleop", display)
        cv2.waitKey(1)
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    ok = True
    for side in ("left", "right"):
        s = samples[side]
        if len(s) < 3:
            print(f"[WARN] Not enough '{side}' samples for resting reference ({len(s)}). Re-try.")
            ok = False
            continue
        elbow_mean = np.mean([x[0] for x in s], axis=0)
        wrist_mean = np.mean([x[1] for x in s], axis=0)
        # Median-aggregate optional hand landmarks; keep None if mostly missing
        valid_pk  = [x[2] for x in s if x[2] is not None]
        valid_ix  = [x[3] for x in s if x[3] is not None]
        pinky_mean = np.mean(valid_pk, axis=0) if len(valid_pk) >= 3 else None
        index_mean = np.mean(valid_ix, axis=0) if len(valid_ix) >= 3 else None
        mapper.set_resting_reference(side, wrist_mean, elbow_mean,
                                     index=index_mean, pinky=pinky_mean)
        print(f"[TELEOP] '{side}' resting reference set (n={len(s)}, "
              f"roll={'yes' if pinky_mean is not None else 'no — hand not visible'}).")
    return ok


# ── Main runtime loop ──────────────────────────────────────────────────────────

def run(
    cap_left: Any,
    cap_right: Any,
    map1x: Any, map1y: Any, map2x: Any, map2y: Any,
    Q: np.ndarray,
    stereo: Any,
    joint_callback: Callable[[str, np.ndarray], None],
    calibration_path: str,
    mono: bool = False,
    draw: bool = True,
    ema_alpha: float = 0.35,
) -> None:
    """
    Main teleop loop. Runs until the user presses 'q' or ESC.

    joint_callback(side, angles_norm) is called every frame with:
        side        : "left" or "right"
        angles_norm : np.ndarray shape (6,) normalized in [-1, 1]
    """
    model_path = _ensure_model()
    mapper     = CameraArmMapper.from_calibration(calibration_path)
    ema: Dict[str, EMAFilter] = {
        "left":  EMAFilter(alpha=ema_alpha),
        "right": EMAFilter(alpha=ema_alpha),
    }

    base_options = _mp_python.BaseOptions(model_asset_path=str(model_path))
    lm_options   = _mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=_mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    with _mp_vision.PoseLandmarker.create_from_options(lm_options) as landmarker:
        ref_ok = capture_resting_reference(
            cap_left, cap_right,
            map1x, map1y, map2x, map2y, Q,
            stereo, landmarker, mapper,
            mono=mono,
        )
        if not ref_ok:
            print("[ERROR] Resting reference capture failed. Exiting.")
            return

        print("[TELEOP] Running — press 'q' / ESC to stop, 'r' to re-capture resting pose.")

        while True:
            if mono:
                ret, frame = cap_left.read()
                if not ret:
                    continue
                rectL: np.ndarray = frame
                points_3d: Optional[np.ndarray] = None
            else:
                cap_left.grab()
                cap_right.grab()
                retL, frameL = cap_left.retrieve()
                retR, frameR = cap_right.retrieve()
                if not retL or not retR:
                    continue
                rectL     = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
                rectR     = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)
                grayL     = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
                grayR     = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
                disp      = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
                points_3d = cv2.reprojectImageTo3D(disp, Q)

            img_h, img_w = rectL.shape[:2]
            rgb      = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, get_monotonic_ts())

            display = rectL.copy()

            if result.pose_landmarks and result.pose_world_landmarks:
                norm_lms  = result.pose_landmarks[0]
                world_lms = result.pose_world_landmarks[0]

                if draw:
                    _draw_pose(display, norm_lms)

                for side, (sh_i, el_i, wr_i, pk_i, idx_i) in _SIDE_INDICES.items():
                    s  = _best_3d(norm_lms[sh_i],  world_lms[sh_i],  img_w, img_h, points_3d)
                    e  = _best_3d(norm_lms[el_i],  world_lms[el_i],  img_w, img_h, points_3d)
                    w  = _best_3d(norm_lms[wr_i],  world_lms[wr_i],  img_w, img_h, points_3d)
                    pk = _best_3d(norm_lms[pk_i],  world_lms[pk_i],  img_w, img_h, points_3d)
                    ix = _best_3d(norm_lms[idx_i], world_lms[idx_i], img_w, img_h, points_3d)

                    if s is None or e is None or w is None:
                        continue

                    raw = mapper.predict(side, s, e, w, index=ix, pinky=pk)
                    if raw is None:
                        continue

                    smooth = ema[side].update(raw)
                    joint_callback(side, smooth)

                    # Overlay current normalised joint values near the wrist
                    wr_lm = norm_lms[_L_WRIST if side == "left" else _R_WRIST]
                    wx, wy = int(wr_lm.x * img_w), int(wr_lm.y * img_h)
                    cv2.putText(
                        display,
                        f"{side[0].upper()}:{smooth[0]:.2f},{smooth[1]:.2f},{smooth[2]:.2f}",
                        (wx, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                    )

            cv2.imshow("Teleop", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                print("[TELEOP] Re-capturing resting reference...")
                for f in ema.values():
                    f.reset()
                capture_resting_reference(
                    cap_left, cap_right,
                    map1x, map1y, map2x, map2y, Q,
                    stereo, landmarker, mapper,
                    mono=mono,
                )