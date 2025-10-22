# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""終止條件定義"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.5) -> torch.Tensor:
    """終止條件：到達目標

    當機器人到達目標範圍內時終止回合
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]

    # 獲取機器人位置
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w

    # 計算距離
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)

    # 到達目標
    reached = distance < threshold

    return reached


def collision_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    collision_threshold: float = 0.3,
) -> torch.Tensor:
    """終止條件：碰撞

    當 LiDAR 偵測到非常近的障礙物時判定為碰撞並終止
    """
    # 獲取 LiDAR 感測器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data

    # 獲取距離數據（兼容多版本 API）
    if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
        # Isaac Sim 5.0+ / Isaac Lab 2025+：需手動計算距離
        hit_points = data.ray_hits_w  # (num_envs, num_rays, 3)
        sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        distances = torch.norm(hit_points - sensor_pos, dim=-1)  # (num_envs, num_rays)
    elif hasattr(data, "hit_distances"):
        distances = data.hit_distances.squeeze(-1)  # Isaac Lab 2025.1
    elif hasattr(data, "distances"):
        distances = data.distances.squeeze(-1)  # Isaac Lab 2024.1
    elif hasattr(data, "ray_distances"):
        distances = data.ray_distances.squeeze(-1)  # Isaac Lab ≤ 2023.1
    else:
        raise AttributeError(f"RayCasterData has no recognized distance attribute")
    
    distances = distances.squeeze(-1) if distances.dim() > 2 else distances  # 確保 (num_envs, num_rays)

    # 檢查是否有射線距離小於閾值
    min_distance, _ = torch.min(distances, dim=-1)
    collision = min_distance < collision_threshold

    return collision



