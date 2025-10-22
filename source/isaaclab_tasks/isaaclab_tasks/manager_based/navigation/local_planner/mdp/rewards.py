# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""獎勵函數定義 - 導航任務獎勵"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_to_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """獎勵：接近目標的進度

    計算當前步與上一步的目標距離差異，獎勵接近目標的行為
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]

    # 獲取機器人位置
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w

    # 計算當前距離
    current_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)

    # 計算進度（需要在環境中儲存上一步距離，這裡簡化為當前距離的負值）
    # 實際應用中，您可以在 env 中加入 self.previous_distance 屬性
    # 這裡使用簡化版本：距離越小，獎勵越大
    reward = -current_distance

    return reward


def reached_goal_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.5) -> torch.Tensor:
    """獎勵：到達目標

    當機器人到達目標範圍內時給予大量獎勵
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]

    # 獲取機器人位置
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w

    # 計算距離
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)

    # 到達目標獎勵
    reward = (distance < threshold).float()

    return reward


def obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.0,
) -> torch.Tensor:
    """懲罰：過於接近障礙物

    當 LiDAR 偵測到障礙物在安全距離內時給予懲罰
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

    # 計算每個環境中最近的障礙物距離
    min_distance, _ = torch.min(distances, dim=-1)

    # 如果最近距離小於安全距離，給予懲罰
    penalty = torch.clamp(safe_distance - min_distance, min=0.0, max=safe_distance)

    return -penalty


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    collision_threshold: float = 0.3,
) -> torch.Tensor:
    """懲罰：碰撞

    當 LiDAR 偵測到非常近的障礙物時判定為碰撞
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
    collision = (min_distance < collision_threshold).float()

    return -collision


def action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """懲罰：動作變化率（鼓勵平滑控制）

    計算當前動作與上一步動作的 L2 距離
    """
    # 注意：這需要環境儲存上一步的動作
    # 簡化版本：懲罰大的角速度
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b

    # 懲罰大的角速度（平方）
    penalty = torch.sum(ang_vel**2, dim=-1)

    return -penalty


def standstill_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """懲罰：靜止不動

    懲罰機器人速度過小的情況
    """
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_b

    # 計算速度大小
    speed = torch.norm(lin_vel[:, :2], dim=-1)

    # 如果速度很小，給予懲罰
    penalty = torch.exp(-10.0 * speed)  # 速度越小，懲罰越大

    return -penalty



