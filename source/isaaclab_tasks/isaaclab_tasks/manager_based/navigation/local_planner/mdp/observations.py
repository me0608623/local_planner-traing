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
    # print("RayCasterData fields:", list(data.__dict__.keys()))  # 已註解：訓練時不需要此調試訊息

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


def predicted_obstacle_distances(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    prediction_horizon: int = 3
) -> torch.Tensor:
    """【PCCBF 啟發】預測未來障礙物距離
    
    設計理念：
    基於 PCCBF-MPC 論文的「前瞻時域地圖（FTD Map）」概念，
    預測未來 N 步機器人周圍的障礙物分布。這讓 Agent 能「提前」
    看到潛在危險，而不是只看當前 LiDAR 數據。
    
    簡化實作：
    - 使用當前 LiDAR + 機器人速度進行線性預測
    - 真實 PCCBF 用卡爾曼濾波器預測動態障礙物，這裡簡化為等速模型
    - 返回「最小預測距離」作為風險指標
    
    Args:
        env: 環境物件
        sensor_cfg: LiDAR 感測器配置
        prediction_horizon: 預測未來幾步（默認3步，約 0.12 秒）
    
    Returns:
        shape (num_envs, 1) 的最小預測距離
        值越小 → 未來碰撞風險越高
    
    教學：為什麼這樣設計？
    - PCCBF 論文強調「預測」比「反應」更安全
    - 在高速運動中，只看當前 LiDAR 會來不及反應
    - 這個觀測讓 Agent 學會「預判」而非「臨時躲避」
    """
    # 獲取當前 LiDAR 距離
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    
    # 計算當前距離（使用 Isaac Sim 5.0 API）
    if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
        hit_points = data.ray_hits_w  # (num_envs, num_rays, 3)
        sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        current_distances = torch.norm(hit_points - sensor_pos, dim=-1)  # (num_envs, num_rays)
    else:
        raise AttributeError("需要 Isaac Sim 5.0+ 的 ray_hits_w API")
    
    # 獲取機器人當前速度（用於預測未來位置）
    robot = env.scene["robot"]
    robot_vel = robot.data.root_lin_vel_b  # (num_envs, 3) 機器人座標系
    
    # 計算時間步長：physics_dt * decimation
    # 例如：0.01s * 4 = 0.04s per RL step
    if hasattr(env, 'step_dt'):
        dt = env.step_dt
    elif hasattr(env, 'physics_dt') and hasattr(env, 'decimation'):
        dt = env.physics_dt * env.decimation
    else:
        # 回退：使用典型值 0.04s (10ms physics @ 4x decimation)
        dt = 0.04
    
    # 簡化預測模型：假設機器人等速直線運動 prediction_horizon 步
    # 真實 PCCBF 會用卡爾曼濾波器，這裡用線性外推
    time_horizon = dt * prediction_horizon  # 總預測時間
    predicted_displacement = robot_vel[:, :2] * time_horizon  # (num_envs, 2) 只考慮 x, y
    
    # 計算位移距離（機器人會移動多遠）
    displacement_norm = torch.norm(predicted_displacement, dim=-1)  # (num_envs,)
    
    # 預測未來位置的障礙物距離
    # 簡化：假設障礙物靜止，計算「如果機器人移動到預測位置」的最小距離
    # 邏輯：當前最小距離 - 機器人會移動的距離 = 預測最小距離
    min_current_distance = torch.min(current_distances, dim=-1)[0]  # (num_envs,)
    predicted_min_distance = min_current_distance - displacement_norm
    
    # 確保非負且處理 NaN
    predicted_min_distance = torch.clamp(predicted_min_distance, min=0.0, max=100.0)
    predicted_min_distance = torch.nan_to_num(predicted_min_distance, nan=10.0, posinf=10.0, neginf=0.0)
    
    return predicted_min_distance.unsqueeze(-1)  # (num_envs, 1)



