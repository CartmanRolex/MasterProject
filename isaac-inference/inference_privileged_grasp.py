import argparse
import logging
import math
import signal
import threading
import time

import torch
from lerobot.envs.factory import make_env

from robot_utils import SO101_FOLLOWER_USD_JOINT_LIMITS


# ==========================================
# 1. Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORANGE_NAMES = ("Orange001", "Orange002", "Orange003")
ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# Continuous gripper joint target in LeIsaac sim radians.
GRIPPER_OPEN = 1.00

# Cartesian target: same XY as orange COM, fixed height above COM.
ABOVE_COM_HEIGHT = 0.08
WARMUP_STEPS = 2

DLS_LAMBDA = 0.04
DEFAULT_SPEED_SCALE = 1.0
BASE_CARTESIAN_STEP = 0.0125
BASE_ROTATION_STEP = 0.06
BASE_JOINT_STEP = 0.0225
AXIS_ALIGNMENT_GAIN = 0.35
RETREAT_ORIENTATION_GAIN = 1.4
ORIENT_DOWN_TOL = 0.03
ORIENT_DOWN_STABLE_STEPS = 8
ORIENT_DOWN_MAX_STEPS = 250
RETREAT_DISTANCE = 0.1
RETREAT_MAX_STEPS = 250
GRIPPER_LOCAL_BACKWARDS_AXIS = (0.0, 0.0, 1.0)
GRIPPER_LOCAL_DOWN_AXIS = (0.0, 0.0, -1.0)
WORLD_DOWN_AXIS = (0.0, 0.0, -1.0)
DEBUG_DRAW_CONTROL_POINTS = True


def tensor_to_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[1:] = -result[1:]
    return result


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vec by quat. Quaternion convention is Isaac's wxyz."""
    quat = quat / (torch.linalg.norm(quat) + 1e-8)
    pure_vec = torch.cat((torch.zeros(1, device=vec.device, dtype=vec.dtype), vec))
    return quat_mul(quat_mul(quat, pure_vec), quat_conjugate(quat))[1:]


def down_axis_error(current_quat_w: torch.Tensor) -> torch.Tensor:
    """World-frame angular error that points the gripper's local down axis at world down.

    This intentionally does not command a full end-effector orientation. Yaw around
    the down axis is left free, so orientation control does not fight the direct
    world-XYZ path to the orange.
    """
    local_down = torch.tensor(
        GRIPPER_LOCAL_DOWN_AXIS,
        device=current_quat_w.device,
        dtype=current_quat_w.dtype,
    )
    world_down = torch.tensor(
        WORLD_DOWN_AXIS,
        device=current_quat_w.device,
        dtype=current_quat_w.dtype,
    )
    current_down = quat_apply(current_quat_w, local_down)
    current_down = current_down / (torch.linalg.norm(current_down) + 1e-8)
    world_down = world_down / (torch.linalg.norm(world_down) + 1e-8)
    return torch.cross(current_down, world_down, dim=0)


def quat_orientation_error(current_quat_w: torch.Tensor, target_quat_w: torch.Tensor) -> torch.Tensor:
    """World-frame rotation vector that rotates current orientation to target orientation."""
    current_quat_w = current_quat_w / (torch.linalg.norm(current_quat_w) + 1e-8)
    target_quat_w = target_quat_w / (torch.linalg.norm(target_quat_w) + 1e-8)
    error_quat = quat_mul(target_quat_w, quat_conjugate(current_quat_w))
    error_quat = error_quat / (torch.linalg.norm(error_quat) + 1e-8)
    if error_quat[0].item() < 0.0:
        error_quat = -error_quat

    vector = error_quat[1:]
    vector_norm = torch.linalg.norm(vector)
    if vector_norm.item() < 1e-8:
        return torch.zeros_like(vector)

    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(error_quat[0], -1.0, 1.0))
    return vector / (vector_norm + 1e-8) * angle


class ResetController:
    """Small keyboard listener: 'r' resets the episode, 's' or 'stop' exits."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_requested = False
        self._stop_requested = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()

    def get_and_clear_reset(self) -> bool:
        with self._lock:
            flag = self._reset_requested
            self._reset_requested = False
            return flag

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _listen(self):
        while True:
            try:
                raw = input()
            except EOFError:
                break

            raw = raw.strip().lower()
            if raw in ("r", "reset"):
                with self._lock:
                    self._reset_requested = True
            elif raw in ("s", "stop"):
                with self._lock:
                    self._stop_requested = True
                print("\nStop requested - will exit after the current step.")


class PrivilegedGraspController:
    """Privileged alignment controller that outputs 6D joint targets.

    It drives the IK EE point above the orange COM in world XY, keeps a fixed
    height above the COM, and aligns the gripper downward.
    """

    def __init__(self, target_orange: str, speed_scale: float):
        if speed_scale <= 0.0:
            raise ValueError(f"speed_scale must be positive, got {speed_scale}")

        self.target_orange = target_orange
        self.speed_scale = speed_scale
        self.max_cartesian_step = BASE_CARTESIAN_STEP * speed_scale
        self.max_rotation_step = BASE_ROTATION_STEP * speed_scale
        self.max_joint_step = BASE_JOINT_STEP * speed_scale
        timeout_scale = DEFAULT_SPEED_SCALE / speed_scale
        self.retreat_max_steps = max(1, math.ceil(RETREAT_MAX_STEPS * timeout_scale))
        self.orient_down_max_steps = max(1, math.ceil(ORIENT_DOWN_MAX_STEPS * timeout_scale))
        self.done = False

        self._joint_ids = None
        self._body_idx = None
        self._jacobi_body_idx = None
        self._jacobi_joint_ids = None
        self._joint_limits = None
        self._origin_w = None
        self._target_com_env = None
        self._phase = "RETREAT"
        self._phase_step = 0
        self._orient_stable_steps = 0
        self._retreat_start_pos_w = None
        self._retreat_start_quat_w = None
        self._debug_legend_printed = False

    def reset(self):
        self.done = False
        self._origin_w = None
        self._target_com_env = None
        self._phase = "RETREAT"
        self._phase_step = 0
        self._orient_stable_steps = 0
        self._retreat_start_pos_w = None
        self._retreat_start_quat_w = None
        self._debug_legend_printed = False

    def compute_action(self, env, current_joint_pos: torch.Tensor) -> torch.Tensor:
        """Return a 6D LeIsaac joint-position target in radians."""
        self._ensure_robot_handles(env)

        self._target_com_env = self._orange_com_env(env, self.target_orange).clone()
        if self._origin_w is None:
            self._origin_w = env.scene.env_origins[0].to(self._target_com_env.device).clone()
            print(
                f"Target {self.target_orange} root_com_pos_w: "
                f"x={self._target_com_env[0].item():+.4f}, "
                f"y={self._target_com_env[1].item():+.4f}, "
                f"z={self._target_com_env[2].item():+.4f}"
            )

        if self._phase == "RETREAT" and self._retreat_start_pos_w is None:
            self._retreat_start_pos_w = self._ee_pos_w(env).clone()
            self._retreat_start_quat_w = self._ee_quat_w(env).clone()

        self._update_phase(env)
        final_target_w, step_target_w = self._compute_targets_w(env)
        self._draw_control_points(env, final_target_w, step_target_w)

        arm_target = self._ik_arm_target(env, current_joint_pos, step_target_w)
        gripper_target = self._gripper_target(current_joint_pos[-1])

        action = current_joint_pos.clone()
        action[:5] = arm_target
        action[5] = gripper_target
        action = self._clamp_joint_limits(action)

        return action

    def _compute_targets_w(self, env):
        """Return (final_target_w, step_target_w) for the current phase.

        final_target_w is the conceptual goal (viz only); None when there is
        no real anchored goal (RETREAT/TILT are pure relative-step control).
        step_target_w is the one-step clamped command handed to IK.
        """
        ee_pos_w = self._ee_pos_w(env)
        ee_quat_w = self._ee_quat_w(env)

        if self._phase == "RETREAT":
            local_back = torch.tensor(
                GRIPPER_LOCAL_BACKWARDS_AXIS,
                device=ee_quat_w.device,
                dtype=ee_quat_w.dtype,
            )
            backwards_dir_w = quat_apply(ee_quat_w, local_back)
            backwards_dir_w = backwards_dir_w / (torch.linalg.norm(backwards_dir_w) + 1e-8)
            step_target_w = ee_pos_w + self.max_cartesian_step * backwards_dir_w
            return None, step_target_w

        if self._phase == "TILT":
            return None, ee_pos_w.clone()

        # MOVE_TO_TARGET
        origin = env.scene.env_origins[0].to(self._target_com_env.device)
        final_target_w = self._target_com_env + origin
        final_target_w = final_target_w.clone()
        final_target_w[2] += ABOVE_COM_HEIGHT
        pos_error = final_target_w - ee_pos_w
        err_norm = torch.linalg.norm(pos_error)
        if err_norm > self.max_cartesian_step:
            pos_error = pos_error / (err_norm + 1e-8) * self.max_cartesian_step
        step_target_w = ee_pos_w + pos_error
        return final_target_w, step_target_w

    def mark_timeout(self):
        self.done = True

    def _ensure_robot_handles(self, env):
        if self._joint_ids is not None:
            return

        robot = env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(list(ARM_JOINT_NAMES))
        if len(joint_ids) != len(ARM_JOINT_NAMES):
            raise RuntimeError(f"Expected arm joints {ARM_JOINT_NAMES}, got {joint_names}")

        body_ids, body_names = robot.find_bodies("gripper")
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one gripper body, got {body_names}")

        self._joint_ids = list(joint_ids)
        self._body_idx = int(body_ids[0])
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [idx + 6 for idx in self._joint_ids]

        limits = []
        for name in list(ARM_JOINT_NAMES) + ["gripper"]:
            lo_deg, hi_deg = SO101_FOLLOWER_USD_JOINT_LIMITS[name]
            limits.append((lo_deg * torch.pi / 180.0, hi_deg * torch.pi / 180.0))
        self._joint_limits = torch.tensor(limits, device=env.device, dtype=torch.float32)

        print(f"IK joints: {joint_names}")
        print(f"IK body:   {body_names[0]}")

    def _orange_com_env(self, env, name: str) -> torch.Tensor:
        origin = env.scene.env_origins[0]
        return env.scene[name].data.root_com_pos_w[0] - origin

    def _ee_pos_w(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        return robot.data.body_pos_w[0, self._body_idx]

    def _ee_quat_w(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        return robot.data.body_quat_w[0, self._body_idx]

    def _update_phase(self, env):
        if self._phase == "RETREAT":
            ee_pos_w = self._ee_pos_w(env)
            displacement = torch.linalg.norm(ee_pos_w - self._retreat_start_pos_w).item()
            self._phase_step += 1
            if displacement >= RETREAT_DISTANCE or self._phase_step >= self.retreat_max_steps:
                self._phase = "TILT"
                self._phase_step = 0
                self._orient_stable_steps = 0
                print(f"RETREAT complete (displacement={displacement:.4f} m); tilting in place.")
            return

        if self._phase == "TILT":
            rot_error_norm = torch.linalg.norm(down_axis_error(self._ee_quat_w(env))).item()
            if rot_error_norm <= ORIENT_DOWN_TOL:
                self._orient_stable_steps += 1
            else:
                self._orient_stable_steps = 0

            self._phase_step += 1
            if (
                self._orient_stable_steps >= ORIENT_DOWN_STABLE_STEPS
                or self._phase_step >= self.orient_down_max_steps
            ):
                self._phase = "MOVE_TO_TARGET"
                self._phase_step = 0
                print(
                    "TILT complete "
                    f"(axis error={rot_error_norm:.4f}); moving above live orange COM."
                )
            return

    def _ik_arm_target(
        self,
        env,
        current_joint_pos: torch.Tensor,
        step_target_w: torch.Tensor,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        ee_pos_w = self._ee_pos_w(env)
        ee_quat_w = self._ee_quat_w(env)

        pos_error = step_target_w - ee_pos_w
        err_norm = torch.linalg.norm(pos_error)
        if err_norm > self.max_cartesian_step:
            pos_error = pos_error / (err_norm + 1e-8) * self.max_cartesian_step

        jacobian = robot.root_physx_view.get_jacobians()[0, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        jacobian_pos = jacobian[0:3]
        jacobian_rot = jacobian[3:6]

        pinv_primary = self._dls_pinv(jacobian_pos)
        delta_pos = pinv_primary @ pos_error.to(jacobian.device)

        if self._phase == "RETREAT":
            if self._retreat_start_quat_w is None:
                self._retreat_start_quat_w = ee_quat_w.clone()
            rot_error = quat_orientation_error(ee_quat_w, self._retreat_start_quat_w)
            rot_norm = torch.linalg.norm(rot_error)
            if rot_norm > self.max_rotation_step:
                rot_error = rot_error / (rot_norm + 1e-8) * self.max_rotation_step

            nullspace = torch.eye(
                len(self._jacobi_joint_ids),
                device=jacobian.device,
                dtype=jacobian.dtype,
            ) - pinv_primary @ jacobian_pos
            jacobian_rot_null = jacobian_rot @ nullspace
            delta_rot = nullspace @ (
                self._dls_pinv(jacobian_rot_null) @ rot_error.to(jacobian.device)
            )
            delta = delta_pos + RETREAT_ORIENTATION_GAIN * delta_rot
        else:
            rot_error = down_axis_error(ee_quat_w)
            rot_norm = torch.linalg.norm(rot_error)
            if rot_norm > self.max_rotation_step:
                rot_error = rot_error / (rot_norm + 1e-8) * self.max_rotation_step

            nullspace = torch.eye(
                len(self._jacobi_joint_ids),
                device=jacobian.device,
                dtype=jacobian.dtype,
            ) - pinv_primary @ jacobian_pos
            jacobian_rot_null = jacobian_rot @ nullspace
            delta_rot = nullspace @ (
                self._dls_pinv(jacobian_rot_null) @ rot_error.to(jacobian.device)
            )
            delta = delta_pos + AXIS_ALIGNMENT_GAIN * delta_rot

        delta = torch.clamp(delta, -self.max_joint_step, self.max_joint_step)

        current_arm_pos = current_joint_pos[:5].to(delta.device)
        arm_target = current_arm_pos + delta
        arm_limits = self._joint_limits[:5].to(delta.device)
        arm_target = torch.max(torch.min(arm_target, arm_limits[:, 1]), arm_limits[:, 0])
        return arm_target

    def _dls_pinv(self, jacobian: torch.Tensor) -> torch.Tensor:
        damping = (DLS_LAMBDA ** 2) * torch.eye(
            jacobian.shape[0],
            device=jacobian.device,
            dtype=jacobian.dtype,
        )
        return jacobian.transpose(0, 1) @ torch.linalg.inv(jacobian @ jacobian.transpose(0, 1) + damping)

    def _gripper_target(self, current_gripper: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(GRIPPER_OPEN, device=current_gripper.device, dtype=current_gripper.dtype)

    def _clamp_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        limits = self._joint_limits.to(action.device)
        return torch.max(torch.min(action, limits[:, 1]), limits[:, 0])

    def _draw_control_points(self, env, final_target_w, step_target_w: torch.Tensor):
        """Draw the world-frame points used by this controller in the Isaac viewport.

        `final_target_w` is optional: RETREAT and TILT have no real anchored
        goal (relative-step control), so only the magenta arrow renders there.
        """
        if not DEBUG_DRAW_CONTROL_POINTS:
            return
        try:
            import carb
            import omni.debugdraw
        except Exception:
            return

        draw = omni.debugdraw.get_debug_draw_interface()
        if hasattr(draw, "clear_points"):
            draw.clear_points()
        if hasattr(draw, "clear_lines"):
            draw.clear_lines()
        ee_pos_w = self._ee_pos_w(env)

        if not self._debug_legend_printed:
            print(
                "Viewport debug: magenta arrow = EE -> per-step IK target (all phases); "
                "cyan dot + dashed orange = final target & alignment line (MOVE_TO_TARGET only)"
            )
            self._debug_legend_printed = True

        def point(pos):
            return carb.Float3(pos[0].item(), pos[1].item(), pos[2].item())

        step_target_w = step_target_w.to(ee_pos_w.device)

        MAGENTA = 0xFFFF00FF
        CYAN = 0xFF00FFFF
        ORANGE = 0xFFFF8800

        if final_target_w is not None:
            final_target_w = final_target_w.to(ee_pos_w.device)
            origin = env.scene.env_origins[0].to(ee_pos_w.device)
            orange_com_w = self._target_com_env.to(ee_pos_w.device) + origin
            DASH_COUNT = 12
            seg = (final_target_w - orange_com_w) / DASH_COUNT
            for i in range(0, DASH_COUNT, 2):
                a = orange_com_w + seg * i
                b = orange_com_w + seg * (i + 1)
                draw.draw_line(point(a), ORANGE, 2.0, point(b), ORANGE, 2.0)
            if hasattr(draw, "draw_point"):
                draw.draw_point(point(final_target_w), CYAN, 12.0)

        # Magenta arrow: shaft EE -> step_target, plus two arrowhead wings.
        shaft_vec = step_target_w - ee_pos_w
        shaft_len = torch.linalg.norm(shaft_vec)
        if shaft_len.item() > 1e-6:
            shaft_dir = shaft_vec / shaft_len
            ref = torch.tensor([0.0, 0.0, 1.0], device=shaft_dir.device, dtype=shaft_dir.dtype)
            if torch.abs(torch.dot(shaft_dir, ref)).item() > 0.95:
                ref = torch.tensor([1.0, 0.0, 0.0], device=shaft_dir.device, dtype=shaft_dir.dtype)
            perp = torch.cross(shaft_dir, ref, dim=0)
            perp = perp / (torch.linalg.norm(perp) + 1e-8)
            head_len = min(0.25 * shaft_len.item(), 0.01)
            head_back = step_target_w - shaft_dir * head_len
            wing1 = head_back + perp * (head_len * 0.5)
            wing2 = head_back - perp * (head_len * 0.5)
            draw.draw_line(point(ee_pos_w), MAGENTA, 5.0, point(step_target_w), MAGENTA, 5.0)
            draw.draw_line(point(step_target_w), MAGENTA, 5.0, point(wing1), MAGENTA, 5.0)
            draw.draw_line(point(step_target_w), MAGENTA, 5.0, point(wing2), MAGENTA, 5.0)


def hold_current_action(obs) -> torch.Tensor:
    return obs["policy"]["joint_pos"][0].to(DEVICE).clone()


def parse_args():
    parser = argparse.ArgumentParser(description="Privileged EE alignment controller for SO101 pick-orange.")
    parser.add_argument("--runs", type=int, default=20, help="Number of alignment runs.")
    parser.add_argument("--max_steps", type=int, default=700, help="Maximum steps per run.")
    parser.add_argument("--target_orange", type=str, default="Orange001", choices=ORANGE_NAMES)
    parser.add_argument(
        "--speed_scale",
        type=float,
        default=DEFAULT_SPEED_SCALE,
        help="Multiplier for Cartesian, rotation, and joint step caps. Lower is slower.",
    )
    parser.add_argument(
        "--no_real_time",
        action="store_true",
        help="Run env steps as fast as possible instead of pacing them to simulation time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading LeIsaac environment...")
    envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)
    suite_name = next(iter(envs_dict))
    env = envs_dict[suite_name][0].envs[0].unwrapped
    env.cfg.episode_length_s = args.max_steps * env.cfg.sim.dt * env.cfg.decimation
    step_dt = env.cfg.sim.dt * env.cfg.decimation
    print(
        "Speed caps: "
        f"scale={args.speed_scale:g}, "
        f"cart={BASE_CARTESIAN_STEP * args.speed_scale:.6f} m/step, "
        f"rot={BASE_ROTATION_STEP * args.speed_scale:.6f} rad/step, "
        f"joint={BASE_JOINT_STEP * args.speed_scale:.6f} rad/step"
    )
    if not args.no_real_time:
        print(f"Real-time pacing enabled ({step_dt:.6f} s/step). Use --no_real_time to run flat out.")

    logging.getLogger("omni").setLevel(logging.ERROR)
    logging.getLogger("carb").setLevel(logging.ERROR)
    try:
        import carb

        carb.settings.get_settings().set_string("/log/level", "error")
    except ImportError:
        pass

    reset_controller = ResetController()
    reset_controller.start()

    completed_runs = 0
    interrupted = False

    def _shutdown_handler(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        for run_idx in range(args.runs):
            print(f"\n{'-' * 52}")
            print(f"Privileged alignment run {run_idx + 1} / {args.runs} | target {args.target_orange}")
            print(f"{'-' * 52}")

            obs, _ = env.reset()
            controller = PrivilegedGraspController(args.target_orange, args.speed_scale)
            controller.reset()

            step_count = 0
            done = False
            next_step_time = time.perf_counter()

            while not done:
                if reset_controller.stop_requested:
                    interrupted = True
                    done = True
                    break

                if reset_controller.get_and_clear_reset():
                    print("\nManual reset requested.")
                    done = True
                    break

                current_joint_pos = hold_current_action(obs)
                if step_count < WARMUP_STEPS:
                    action = current_joint_pos
                else:
                    action = controller.compute_action(env, current_joint_pos)

                if action.shape != (6,):
                    raise RuntimeError(f"Controller produced action shape {tuple(action.shape)}, expected (6,)")

                obs, _reward, terminated, truncated, _info = env.step(action.unsqueeze(0))
                step_count += 1

                if not args.no_real_time:
                    next_step_time += step_dt
                    sleep_time = next_step_time - time.perf_counter()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    else:
                        next_step_time = time.perf_counter()

                if not done and step_count >= args.max_steps:
                    controller.mark_timeout()
                    completed_runs += 1
                    print(f"Run {run_idx + 1}: alignment run reached max steps ({step_count})")
                    done = True

                done = done or tensor_to_bool(terminated) or tensor_to_bool(truncated)

            if interrupted:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("\n" + "=" * 40)
        print("PRIVILEGED ALIGNMENT SUMMARY")
        print(f"Runs completed: {completed_runs}/{args.runs}")
        print("=" * 40)
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    main()
