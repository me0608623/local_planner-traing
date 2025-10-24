# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""觀測項定義 - LiDAR、機器人狀態、目標資訊"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """LiDAR 距離觀測（兼容 Isaac Lab 2023-2025+）

    Returns:
        LiDAR 點的距離數據，shape (num_envs, num_rays)
    """
    # 獲取 LiDAR 感測器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data

    # 調試：顯示可用的 API 屬性（首次調用時）
    print("RayCasterData fields:", list(data.__dict__.keys()))

    # 嘗試多版本屬性存取（從新到舊）
    if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
        # Isaac Sim 5.0+ / Isaac Lab 2025+：需手動計算距離
        # ray_hits_w: (num_envs, num_rays, 3) - 世界座標中的射線命中點
        # pos_w: (num_envs, 3) - 世界座標中的感測器位置
        hit_points = data.ray_hits_w  # (num_envs, num_rays, 3)
        sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3) 擴展以廣播
        
        # 計算每條射線的距離
        distances = torch.norm(hit_points - sensor_pos, dim=-1)  # (num_envs, num_rays)
        # print("✅ 使用 ray_hits_w API (2025+ / Sim 5.0) - 手動計算距離")
        
    elif hasattr(data, "hit_distances"):
        distances = data.hit_distances  # Isaac Lab 2025.1
        print("✅ 使用 hit_distances API (2025.1)")
    elif hasattr(data, "distances"):
        distances = data.distances  # Isaac Lab 2024.1
        print("✅ 使用 distances API (2024.1)")
    elif hasattr(data, "ray_distances"):
        distances = data.ray_distances  # Isaac Lab ≤ 2023.1
        print("✅ 使用 ray_distances API (2023.1)")
    else:
        raise AttributeError(
            f"RayCasterData has no recognized distance attribute. Available: {list(data.__dict__.keys())}"
        )

    # 正規化到 [0, 1]
    max_distance = sensor.cfg.max_distance
    distances = distances / max_distance
    distances = torch.clamp(distances, 0.0, 1.0)
    
    # 將形狀從 (num_envs, num_rays, 1) 壓縮到 (num_envs, num_rays)
    return distances.squeeze(-1)


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """機器人基座線速度（機器人座標系）

    Returns:
        形狀 (num_envs, 3) 的線速度 [vx, vy, vz]
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """機器人基座角速度（機器人座標系）

    Returns:
        形狀 (num_envs, 3) 的角速度 [wx, wy, wz]
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def goal_position_in_robot_frame(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """目標位置（機器人座標系）

    Returns:
        形狀 (num_envs, 2) 的相對位置 [dx, dy]
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]  # 世界座標系中的目標位置 (x, y, z)

    # 獲取機器人資產
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w  # 世界座標系中的機器人位置
    robot_quat_w = robot.data.root_quat_w  # 世界座標系中的機器人方向

    # 計算相對位置（世界座標系）
    goal_pos_rel_w = goal_pos_w - robot_pos_w

    # 轉換到機器人座標系（使用新版 API）
    goal_pos_rel_b = math_utils.quat_apply_inverse(robot_quat_w, goal_pos_rel_w)

    # 只返回 x, y（忽略 z）
    return goal_pos_rel_b[:, :2]


def distance_to_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """到目標的距離

    Returns:
        形狀 (num_envs, 1) 的距離標量
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]

    # 獲取機器人位置
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w

    # 計算 2D 距離（忽略 z）
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1, keepdim=True)

    return distance



