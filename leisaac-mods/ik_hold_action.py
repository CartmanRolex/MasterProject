"""IK action term that holds joint targets when the command delta is zero.

With Isaac Lab's standard DifferentialInverseKinematicsAction and use_relative_mode=True,
a zero command computes:
    ee_pos_des = current_ee_pos + 0  →  pose_error = 0  →  delta_joint = 0
    joint_targets = current_joint_pos

This means joint targets are always set to the current positions when no delta is applied,
giving the PD controller zero error and therefore zero restoring force against external loads
(e.g., gravity from a grasped object).

This module fixes that by caching the last IK-computed joint targets and reapplying them
whenever the command is zero, so the arm actively holds its last commanded position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialInverseKinematicsHoldAction(DifferentialInverseKinematicsAction):
    """IK action term that holds joint targets when the command delta is zero.

    When the processed command is all zeros, the last computed joint position targets
    are reapplied verbatim, keeping the PD controller error non-zero if the arm has
    drifted. This prevents the arm from sagging under external loads during idle steps.

    Optionally, when ``orientation_lock_enabled`` is set to True by the controlling device,
    a stable EE orientation target (in robot root frame) is maintained across position
    commands. This prevents pitch drift accumulation caused by kinematic coupling: without
    it, ``process_actions()`` resets ``ee_quat_des = current_quat`` each step, making
    rotation_error = 0 and leaving the IK with no force to resist orientation drift.
    """

    cfg: "DifferentialInverseKinematicsHoldActionCfg"

    def __init__(self, cfg: "DifferentialInverseKinematicsHoldActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # Cached joint position targets from the last active (non-zero) IK step.
        # Shape: (num_envs, num_controlled_joints)
        self._held_joint_pos_des: torch.Tensor | None = None
        # Stable EE orientation target in root frame, used when orientation_lock_enabled.
        # Captured on first locked step; updated when a rotation command (LB/LT) is given.
        # Shape: (num_envs, 4) quaternion (w, x, y, z)
        self._target_ee_quat: torch.Tensor | None = None
        # Set to True by the controlling device to activate orientation locking.
        self.orientation_lock_enabled: bool = False

    def apply_actions(self) -> None:
        # Zero-delta: hold cached joint targets (prevents gravity sag).
        if self._held_joint_pos_des is not None and not bool(self._processed_actions.any()):
            self._asset.set_joint_position_target(self._held_joint_pos_des, self._joint_ids)
            return

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()

            if self.orientation_lock_enabled:
                # _processed_actions layout (pose + relative mode): [dx, dy, dz, droll, dpitch, dyaw]
                rot_cmd = self._processed_actions[:, 3:]  # (N, 3)
                if not bool(rot_cmd.any()):
                    # Pure position command: lock to the stable orientation target.
                    # Without this, process_actions() sets ee_quat_des = current_quat each step,
                    # so rotation_error = 0 and kinematic pitch coupling goes uncorrected.
                    if self._target_ee_quat is None:
                        # Capture current orientation as the reference on first locked step.
                        self._target_ee_quat = ee_quat_curr.clone()
                    self._ik_controller.ee_quat_des[:] = self._target_ee_quat
                else:
                    # Rotation command (e.g. LB/LT pitch): adopt the IK's updated quat as
                    # the new stable target so subsequent position commands hold from here.
                    self._target_ee_quat = self._ik_controller.ee_quat_des.clone()
            else:
                # Locking disabled: clear the cached target so re-enabling captures fresh.
                self._target_ee_quat = None

            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        self._held_joint_pos_des = joint_pos_des.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # "Reset all" cases
        if env_ids is None or isinstance(env_ids, slice):
            self._held_joint_pos_des = None
            self._target_ee_quat = None
            return
        if self._held_joint_pos_des is None:
            return
        # Partial reset: re-seed joint targets and orientation target from current state.
        self._held_joint_pos_des[env_ids] = (
            self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        )
        if self._target_ee_quat is not None:
            _, ee_quat_curr = self._compute_frame_pose()
            self._target_ee_quat[env_ids] = ee_quat_curr[env_ids].clone()


@configclass
class DifferentialInverseKinematicsHoldActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`DifferentialInverseKinematicsHoldAction`.

    Identical to :class:`DifferentialInverseKinematicsActionCfg` — just swaps the
    underlying class so that zero-delta steps hold the last joint targets instead of
    chasing the arm's drifting position.
    """

    class_type: type[ActionTerm] = DifferentialInverseKinematicsHoldAction


class RelativeJointPositionHoldAction(RelativeJointPositionAction):
    """Direct joint position action term that holds targets when the command delta is zero.

    With Isaac Lab's standard RelativeJointPositionAction, a zero command computes:
        target = current_joint_pos + 0  →  PD error = 0  →  zero restoring force

    This means joints drift freely under external loads (gravity, cable coupling).
    This class fixes that by caching the last commanded absolute target and reapplying
    it whenever the command is zero, so joints actively hold their last position.
    """

    def __init__(self, cfg: "RelativeJointPositionHoldActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # Cached absolute joint position targets from the last active (non-zero) step.
        # Shape: (num_envs, num_controlled_joints)
        self._held_joint_pos: torch.Tensor | None = None

    def apply_actions(self) -> None:
        # Zero-delta: hold cached joint targets (prevents gravity sag / coupling drift).
        if self._held_joint_pos is not None and not bool(self._processed_actions.any()):
            self._asset.set_joint_position_target(self._held_joint_pos, self._joint_ids)
            return

        # Seed cache from current positions on first call.
        if self._held_joint_pos is None:
            self._held_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()

        # Non-zero delta: integrate into absolute target, cache, and apply.
        self._held_joint_pos = self._held_joint_pos + self._processed_actions
        self._asset.set_joint_position_target(self._held_joint_pos, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            self._held_joint_pos = None
            return
        if self._held_joint_pos is not None:
            self._held_joint_pos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()


@configclass
class RelativeJointPositionHoldActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`RelativeJointPositionHoldAction`.

    Identical to :class:`RelativeJointPositionActionCfg` — just swaps the underlying
    class so that zero-delta steps hold the last joint targets instead of chasing the
    joint's drifting position.
    """

    class_type: type[ActionTerm] = RelativeJointPositionHoldAction
