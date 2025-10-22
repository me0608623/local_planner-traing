# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""動作項定義 - 差速驅動控制"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialDriveAction(ActionTerm):
    """差速驅動動作：接收 [v, ω]，轉換為左右輪速度

    動作空間：[-1, 1]^2
    - action[0]: 正規化線速度 v（前進/後退）
    - action[1]: 正規化角速度 ω（左轉/右轉）
    """

    cfg: DifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 獲取機器人資產
        self._asset = env.scene[cfg.asset_name]

        # 獲取左右輪關節索引（find_joints 返回列表，需轉換為張量）
        left_indices_list, _ = self._asset.find_joints(cfg.left_wheel_joint_names)
        right_indices_list, _ = self._asset.find_joints(cfg.right_wheel_joint_names)
        
        # 轉換為張量
        self._left_wheel_indices = torch.tensor(left_indices_list, dtype=torch.long, device=self._asset.device)
        self._right_wheel_indices = torch.tensor(right_indices_list, dtype=torch.long, device=self._asset.device)

        # 差速驅動參數
        self._wheel_radius = cfg.wheel_radius
        self._wheel_base = cfg.wheel_base
        self._max_v = cfg.max_linear_speed
        self._max_w = cfg.max_angular_speed

    @property
    def action_dim(self) -> int:
        return 2  # [v, ω]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """將 [v, ω] 轉換為左右輪速度指令"""
        # 儲存原始動作
        self._raw_actions = actions.clone()

        # 反正規化：從 [-1, 1] 映射到實際速度範圍
        v = actions[:, 0] * self._max_v  # 線速度 (m/s)
        w = actions[:, 1] * self._max_w  # 角速度 (rad/s)

        # 差速驅動運動學：v = r/2 * (v_L + v_R), ω = r/L * (v_R - v_L)
        # 反解：v_L = (v - ωL/2) / r, v_R = (v + ωL/2) / r
        v_left = (v - w * self._wheel_base / 2.0) / self._wheel_radius
        v_right = (v + w * self._wheel_base / 2.0) / self._wheel_radius

        # 組合輪速度指令（角速度）
        wheel_velocities = torch.zeros((actions.shape[0], len(self._left_wheel_indices) + len(self._right_wheel_indices)), device=actions.device)
        wheel_velocities[:, :len(self._left_wheel_indices)] = v_left.unsqueeze(-1)
        wheel_velocities[:, len(self._left_wheel_indices):] = v_right.unsqueeze(-1)

        self._processed_actions = wheel_velocities

    def apply_actions(self):
        """將輪速度指令應用到關節"""
        # 設定關節速度目標
        all_indices = torch.cat([self._left_wheel_indices, self._right_wheel_indices])
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=all_indices)


from isaaclab.utils import configclass

@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """差速驅動動作配置"""

    class_type: type[ActionTerm] = DifferentialDriveAction

    # 機器人資產名稱
    asset_name: str = ""
    # 左右輪關節名稱
    left_wheel_joint_names: Sequence[str] | None = None
    right_wheel_joint_names: Sequence[str] | None = None
    # 輪子參數
    wheel_radius: float = 0.1
    wheel_base: float = 0.4
    # 速度限制
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0

