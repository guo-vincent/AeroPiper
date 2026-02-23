"""
Simple joint-position GUI (14 sliders) for AeroPiper.

Sliders:
  - 6 for LEFT arm joints
  - 6 for RIGHT arm joints
  - 1 for LEFT gripper (applies to 6 gripper actuators; thumb abduction held at max)
  - 1 for RIGHT gripper (applies to 6 gripper actuators; thumb abduction held at max)

This GUI directly sets MuJoCo joint qpos values each frame (good for posing / inspection).

Camera teleoperation is enabled when --cal is provided.
Use --mono for a single webcam (no stereo depth). Omit --cal to run GUI-only.

# Single camera:
python teleop/camera_control.py --cal teleop/camera_module/camera_joint_calibration.json --mono

# Full stereo:
python teleop/camera_control.py --cal teleop/camera_module/camera_joint_calibration.json --maps calibration/rectification/rect_maps.npz --rect calibration/rectification/rectification.yaml

# GUI only, no camera:
python teleop/camera_control.py
"""

import argparse
from pathlib import Path
import sys
import signal

import numpy as np
import cv2

from typing import TYPE_CHECKING, Any, Callable, List

if TYPE_CHECKING:
    import tkinter as tk

# Allow running as `python teleop/gui.py` (ensure repo root is on sys.path)
_MODULE_DIR = Path(__file__).resolve().parent
_TELEOP_DIR = _MODULE_DIR.parent
_REPO_ROOT  = _TELEOP_DIR.parent

for p in (_REPO_ROOT, _TELEOP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import aeropiper as suite

import threading
from camera_module.arm_pose_teleop import run as teleop_run


# ── Teleop callback ────────────────────────────────────────────────────────────

def make_callback(
    sim: Any,
    left_vars: "List[tk.DoubleVar]",
    right_vars: "List[tk.DoubleVar]",
    left_joints: List[str],
    right_joints: List[str],
) -> Callable[[str, np.ndarray], None]:
    """
    Returns a thread-safe callback that the teleop loop calls each frame.
    Rescales the regressor's [-1, 1] outputs to each joint's actual radian
    range and writes them into the Tkinter DoubleVars that drive the sliders.
    tick() picks those values up on the next GUI frame.
    """
    def cb(side: str, angles_norm: np.ndarray) -> None:
        joints   = left_joints  if side == "left" else right_joints
        var_list = left_vars    if side == "left" else right_vars
        for var, jname, a in zip(var_list, joints, angles_norm):
            jid = sim.model.joint_name2id(jname)
            lo, hi = sim.model.jnt_range[jid]
            var.set(float(np.clip(lo + (a + 1) / 2 * (hi - lo), lo, hi)))
    return cb


# ── MuJoCo helpers ────────────────────────────────────────────────────────────

def _get_joint_limit(sim, joint_name: str):
    """Returns (lo, hi, limited) for a MuJoCo joint name."""
    jid = sim.model.joint_name2id(joint_name)
    limited = bool(sim.model.jnt_limited[jid])
    lo, hi = sim.model.jnt_range[jid]
    if not limited:
        lo, hi = -np.pi, np.pi
    return float(lo), float(hi), limited


def _intersection_limits(sim, joint_names):
    if not joint_names:
        return -1.0, 1.0
    los, his = [], []
    for j in joint_names:
        lo, hi, _ = _get_joint_limit(sim, j)
        los.append(lo)
        his.append(hi)
    lo_i = max(los)
    hi_i = min(his)
    if lo_i >= hi_i:
        return los[0], his[0]
    return lo_i, hi_i


def _get_actuator_ctrlrange(sim, actuator_name: str):
    aid = sim.model.actuator_name2id(actuator_name)
    lo, hi = sim.model.actuator_ctrlrange[aid]
    return int(aid), float(lo), float(hi)


def _build_gripper_actuator_map(sim, gripper_actuator_names):
    names = list(gripper_actuator_names)
    if not names:
        return None

    def _is_thumb_abd(n: str) -> bool:
        s = n.lower()
        return ("thumb" in s) and ("abd" in s)

    thumb_abd_name = next((n for n in names if _is_thumb_abd(n)), None)
    if thumb_abd_name is None:
        thumb_abd_name = next((n for n in names if "abd" in n.lower()), None)

    thumb_abd = None
    if thumb_abd_name is not None:
        aid, lo, hi = _get_actuator_ctrlrange(sim, thumb_abd_name)
        thumb_abd = (aid, lo, hi)
        names = [n for n in names if n != thumb_abd_name]

    controlled = []
    for n in names:
        aid, lo, hi = _get_actuator_ctrlrange(sim, n)
        controlled.append((aid, lo, hi))

    return {"thumb_abd": thumb_abd, "controlled": controlled}


def _gripper_extra_flex_joints(gripper_joint_names):
    out = []
    for j in gripper_joint_names:
        jl = j.lower()
        if ("_pip" in jl) or ("_dip" in jl) or ("thumb_ip" in jl):
            out.append(j)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PickPlace")

    parser.add_argument(
        "--cal", type=str, default="",
        help="Path to camera_joint_calibration.json. Omit to run GUI without teleop.",
    )
    parser.add_argument(
        "--mono", action="store_true",
        help="Single-camera mode: uses MediaPipe world landmarks, no stereo depth.",
    )
    parser.add_argument("--left",  type=int, default=0, help="Left camera device index.")
    parser.add_argument("--right", type=int, default=1, help="Right camera device index.")
    parser.add_argument("--maps",  type=str, default="", help="NPZ rectification maps (stereo).")
    parser.add_argument("--rect",  type=str, default="", help="YAML with Q matrix (stereo).")
    parser.add_argument("--num_disparities", type=int, default=128)
    parser.add_argument("--blocksize",       type=int, default=5)
    args = parser.parse_args()

    cap_left = cap_right = None
    map1x = map1y = map2x = map2y = Q = stereo = None
    teleop_enabled = bool(args.cal)

    if teleop_enabled:
        cap_left = cv2.VideoCapture(args.left, cv2.CAP_DSHOW)

        if not args.mono:
            if not args.maps or not args.rect:
                raise ValueError(
                    "Stereo mode requires --maps and --rect. "
                    "Use --mono for single-camera operation."
                )
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
            print("[GUI] Stereo teleop mode.")
        else:
            print("[GUI] Mono teleop mode (world landmarks only).")
    else:
        print("[GUI] No --cal provided — slider-only mode, no camera required.")

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as e:
        raise ImportError(
            "tkinter is required for this GUI.\n"
            "On Ubuntu/Debian: sudo apt-get install python3-tk"
        ) from e

    env = suite.make(
        env_name=args.env,
        robots="AeroPiper",
        env_configuration="single-robot",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
        renderer="mjviewer",
        ignore_done=True,
    )
    env.reset()

    sim   = env.sim
    robot = env.robots[0]

    if getattr(env, "viewer", None) is not None:
        if hasattr(env.viewer, "set_camera"):
            env.viewer.set_camera(camera_id=-1)
        if hasattr(env.viewer, "update"):
            env.viewer.update()

    arm_joints = list(robot.robot_model.arm_joints)
    split = int(len(arm_joints) / len(robot.arms))
    right_arm_joints = arm_joints[:split]
    left_arm_joints  = arm_joints[split:]

    left_gripper_joints     = list(robot.gripper["left"].joints)
    right_gripper_joints    = list(robot.gripper["right"].joints)
    left_gripper_actuators  = list(robot.gripper["left"].actuators)
    right_gripper_actuators = list(robot.gripper["right"].actuators)

    left_grip_act_map       = _build_gripper_actuator_map(sim, left_gripper_actuators)
    right_grip_act_map      = _build_gripper_actuator_map(sim, right_gripper_actuators)
    left_grip_extra_joints  = _gripper_extra_flex_joints(left_gripper_joints)
    right_grip_extra_joints = _gripper_extra_flex_joints(right_gripper_joints)

    # --- GUI ---
    root = tk.Tk()
    root.title("AeroPiper Slider Control (14 sliders)")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Title.TLabel",              font=("Arial", 12, "bold"))
    style.configure("Section.TLabelframe.Label", font=("Arial", 10, "bold"))

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)

    vars_by_name = {}

    def add_slider(parent, row, label, lo, hi, initial, *, color: str):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        v = tk.DoubleVar(value=float(initial))
        s = tk.Scale(
            parent,
            from_=lo, to=hi,
            variable=v,
            orient="horizontal",
            resolution=0.001,
            length=360,
            showvalue=False,
            troughcolor=color,
            activebackground=color,
            highlightthickness=0,
            bd=0,
        )
        s.grid(row=row, column=1, sticky="ew", pady=4)
        val_lbl = ttk.Label(parent, text=f"{float(initial): .4f}", width=10)
        val_lbl.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)

        def _update_label(*_):
            val_lbl.configure(text=f"{v.get(): .4f}")
        v.trace_add("write", _update_label)
        vars_by_name[label] = v
        return v

    # Top buttons
    topbar = ttk.Frame(main_frame)
    topbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    topbar.columnconfigure(0, weight=1)
    ttk.Label(topbar, text="AeroPiper control", style="Title.TLabel").grid(row=0, column=0, sticky="w")

    def reset_sliders_to_current_pose():
        for j in left_arm_joints:
            vars_by_name[j].set(float(sim.data.get_joint_qpos(j)))
        for j in right_arm_joints:
            vars_by_name[j].set(float(sim.data.get_joint_qpos(j)))
        if left_grip_act_map and left_grip_act_map["controlled"]:
            aid, lo, hi = left_grip_act_map["controlled"][0]
            cur = float(sim.data.ctrl[aid])
            a = 0.0 if hi <= lo else (cur - lo) / (hi - lo)
            vars_by_name["left_gripper (6 DoF; thumb_abd pinned to max)"].set(float(np.clip(a, 0.0, 1.0)))
        if right_grip_act_map and right_grip_act_map["controlled"]:
            aid, lo, hi = right_grip_act_map["controlled"][0]
            cur = float(sim.data.ctrl[aid])
            a = 0.0 if hi <= lo else (cur - lo) / (hi - lo)
            vars_by_name["right_gripper (6 DoF; thumb_abd pinned to max)"].set(float(np.clip(a, 0.0, 1.0)))

    def focus_free_camera():
        if getattr(env, "viewer", None) is not None:
            if hasattr(env.viewer, "set_camera"):
                env.viewer.set_camera(camera_id=-1)
            if hasattr(env.viewer, "update"):
                env.viewer.update()

    ttk.Button(topbar, text="Reset sliders to current pose", command=reset_sliders_to_current_pose).grid(
        row=0, column=1, sticky="e", padx=(8, 0))
    ttk.Button(topbar, text="Free camera", command=focus_free_camera).grid(
        row=0, column=2, sticky="e", padx=(8, 0))

    left_frame = ttk.Labelframe(main_frame, text="LEFT arm (6 joints)", style="Section.TLabelframe")
    left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
    left_frame.columnconfigure(1, weight=1)

    right_frame = ttk.Labelframe(main_frame, text="RIGHT arm (6 joints)", style="Section.TLabelframe")
    right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
    right_frame.columnconfigure(1, weight=1)

    gripper_frame = ttk.Labelframe(
        main_frame, text="Grippers (2 sliders, each applies to 7 DoFs)",
        style="Section.TLabelframe")
    gripper_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
    gripper_frame.columnconfigure(1, weight=1)

    LEFT_COLOR  = "#4ea1ff"
    RIGHT_COLOR = "#46d18c"
    GRIP_COLOR  = "#ffb347"

    left_joint_vars = []
    for i, j in enumerate(left_arm_joints):
        lo, hi, _ = _get_joint_limit(sim, j)
        q = float(sim.data.get_joint_qpos(j))
        left_joint_vars.append(add_slider(left_frame, i, j, lo, hi, q, color=LEFT_COLOR))

    right_joint_vars = []
    for i, j in enumerate(right_arm_joints):
        lo, hi, _ = _get_joint_limit(sim, j)
        q = float(sim.data.get_joint_qpos(j))
        right_joint_vars.append(add_slider(right_frame, i, j, lo, hi, q, color=RIGHT_COLOR))

    left_grip_var = add_slider(
        gripper_frame, 0, "left_gripper (6 DoF; thumb_abd pinned to max)",
        0.0, 1.0, 0.0, color=GRIP_COLOR)
    right_grip_var = add_slider(
        gripper_frame, 1, "right_gripper (6 DoF; thumb_abd pinned to max)",
        0.0, 1.0, 0.0, color=GRIP_COLOR)

    if teleop_enabled:
        teleop_thread = threading.Thread(
            target=teleop_run,
            kwargs=dict(
                cap_left=cap_left,
                cap_right=cap_right,        # None in mono mode
                map1x=map1x, map1y=map1y,   # None in mono mode
                map2x=map2x, map2y=map2y,   # None in mono mode
                Q=Q,                        # None in mono mode
                stereo=stereo,              # None in mono mode
                mono=args.mono,
                joint_callback=make_callback(
                    sim,
                    left_joint_vars,  right_joint_vars,
                    left_arm_joints,  right_arm_joints,
                ),
                calibration_path=args.cal,
            ),
            daemon=True,
        )
        teleop_thread.start()
        print("[GUI] Teleop thread started — hold resting pose in the Teleop window.")

    # ── Physics / tick loop ───────────────────────────────────────────────────

    running = {"ok": True}

    def on_close():
        running["ok"] = False
        try:
            env.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def _sigint_handler(_sig, _frame):
        if running["ok"]:
            try:
                root.after(0, on_close)
            except Exception:
                on_close()

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        pass

    interp_steps         = 8
    sim_steps_per_interp = 2

    def _set_joint_qvel_zero(joint_name: str):
        addr = sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(addr, (int, np.integer)):
            sim.data.qvel[int(addr)] = 0.0
        else:
            a0, a1 = addr
            sim.data.qvel[a0:a1] = 0.0

    def tick():
        try:
            if not running["ok"]:
                return

            left_targets  = np.array([float(v.get()) for v in left_joint_vars],  dtype=float)
            right_targets = np.array([float(v.get()) for v in right_joint_vars], dtype=float)
            l_grip_alpha  = float(left_grip_var.get())
            r_grip_alpha  = float(right_grip_var.get())

            left_curr  = np.array([float(sim.data.get_joint_qpos(j)) for j in left_arm_joints],  dtype=float)
            right_curr = np.array([float(sim.data.get_joint_qpos(j)) for j in right_arm_joints], dtype=float)

            for k in range(interp_steps):
                a = (k + 1) / interp_steps
                left_next  = left_curr  + a * (left_targets  - left_curr)
                right_next = right_curr + a * (right_targets - right_curr)

                for j, q in zip(left_arm_joints, left_next):
                    sim.data.set_joint_qpos(j, float(q))
                    _set_joint_qvel_zero(j)
                for j, q in zip(right_arm_joints, right_next):
                    sim.data.set_joint_qpos(j, float(q))
                    _set_joint_qvel_zero(j)

                if left_grip_act_map:
                    for aid, lo, hi in left_grip_act_map["controlled"]:
                        sim.data.ctrl[aid] = lo + float(np.clip(l_grip_alpha, 0.0, 1.0)) * (hi - lo)
                    if left_grip_act_map["thumb_abd"] is not None:
                        aid, lo, hi = left_grip_act_map["thumb_abd"]
                        sim.data.ctrl[aid] = hi
                for j in left_grip_extra_joints:
                    lo, hi, _ = _get_joint_limit(sim, j)
                    sim.data.set_joint_qpos(j, lo + float(np.clip(l_grip_alpha, 0.0, 1.0)) * (hi - lo))
                    _set_joint_qvel_zero(j)

                if right_grip_act_map:
                    for aid, lo, hi in right_grip_act_map["controlled"]:
                        sim.data.ctrl[aid] = lo + float(np.clip(r_grip_alpha, 0.0, 1.0)) * (hi - lo)
                    if right_grip_act_map["thumb_abd"] is not None:
                        aid, lo, hi = right_grip_act_map["thumb_abd"]
                        sim.data.ctrl[aid] = hi
                for j in right_grip_extra_joints:
                    lo, hi, _ = _get_joint_limit(sim, j)
                    sim.data.set_joint_qpos(j, lo + float(np.clip(r_grip_alpha, 0.0, 1.0)) * (hi - lo))
                    _set_joint_qvel_zero(j)

                sim.forward()
                for _ in range(sim_steps_per_interp):
                    sim.step()

            if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
                env.viewer.update()

            root.after(20, tick)
        except KeyboardInterrupt:
            on_close()
            return

    tick()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_close()


if __name__ == "__main__":
    main()