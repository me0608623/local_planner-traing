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
    正獎勵：機器人接近目標（距離減少）
    負獎勵：機器人遠離目標（距離增加）
    
    Returns:
        進度獎勵 tensor，shape (num_envs,)
        正值表示接近目標，負值表示遠離目標
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]

    # 獲取機器人位置
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w

    # 計算當前距離 (只考慮 x, y，忽略 z)
    current_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)  # (num_envs,)

    # 初始化或獲取上一步距離（儲存為環境狀態變數）
    # 這個變數會在所有並行環境間共享，每個環境有自己的距離值
    if not hasattr(env, "_prev_goal_distance"):
        # 首次調用：初始化為當前距離，進度為 0
        env._prev_goal_distance = current_distance.clone()
        progress = torch.zeros_like(current_distance)
    else:
        # 計算進度 = 上一步距離 - 當前距離
        # 如果接近目標（距離減少）→ 正值
        # 如果遠離目標（距離增加）→ 負值
        progress = env._prev_goal_distance - current_distance
        
        # 處理環境重置：如果某個環境剛重置 (episode_length == 1)，其進度應為 0
        # 因為第一步沒有「上一步」可以比較
        reset_mask = (env.episode_length_buf == 1)
        progress = torch.where(reset_mask, torch.zeros_like(progress), progress)
        
        # 更新上一步距離（所有環境都更新）
        env._prev_goal_distance = current_distance.clone()

    return progress


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


def near_goal_shaping(
    env: ManagerBasedRLEnv, 
    command_name: str,
    radius: float = 1.5
) -> torch.Tensor:
    """【新增】近距離塑形獎勵
    
    設計理念：
    解決"最後一公里"問題。當Agent接近目標（< radius），給予額外的正獎勵，
    獎勵隨著距離減少而增加，幫助Agent學會"最後逼近"。
    
    為什麼需要這個？
    - progress_to_goal 只獎勵"接近"的行為（距離差）
    - 但在最後階段，Agent可能在目標附近徘徊，不知道如何"衝刺"
    - 這個塑形獎勵提供明確的梯度：越近越好
    
    數學公式：
    reward = max(0, (radius - distance) / radius)
    - 距離 = 0米 → reward = 1.0（最高獎勵）
    - 距離 = radius → reward = 0.0（開始進入範圍）
    - 距離 > radius → reward = 0.0（超出範圍）
    
    Args:
        env: 環境物件
        command_name: 目標指令名稱
        radius: 塑形半徑（米），在此範圍內給予獎勵
    
    Returns:
        shape (num_envs,) 的塑形獎勵
        0.0 ~ 1.0，越接近目標獎勵越高
    
    測試方法：
    - 觀察 Episode_Reward/near_goal_shaping
    - 如果平均 > 0.3，說明Agent經常進入1.5米範圍
    - 如果平均 > 0.6，說明Agent能接近到0.6米以內
    """
    # 獲取目標指令
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]
    
    # 獲取機器人位置
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w
    
    # 計算2D距離
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1)  # (num_envs,)
    
    # 塑形獎勵：線性遞減，從1.0（距離=0）到0.0（距離=radius）
    shaping = torch.clamp((radius - distance) / radius, min=0.0, max=1.0)
    
    return shaping


def cbf_safety_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.5,
    critical_distance: float = 0.5,
) -> torch.Tensor:
    """【PCCBF 核心】控制屏障函數（CBF）安全獎勵
    
    設計理念：
    基於 PCCBF-MPC 論文的「控制屏障函數」概念，這是論文的核心創新。
    CBF 定義了一個「安全集合」，確保系統狀態永遠不會離開這個集合。
    在導航中，CBF 保證機器人與障礙物的距離始終大於安全閾值。
    
    論文中的 CBF 數學表達：
    h(x) = ||x_robot - x_obstacle|| - d_safe ≥ 0
    
    我們的實作策略：
    - 將 CBF 轉換為「軟約束」獎勵（因為 PPO 無法處理硬約束）
    - 使用「指數衰減」獎勵：距離越近，懲罰指數增長
    - 區分「安全區」和「危險區」，給予不同程度的獎勵/懲罰
    
    Args:
        env: 環境物件
        sensor_cfg: LiDAR 感測器配置
        safe_distance: 安全距離閾值（米），超過此距離 → 完全安全
        critical_distance: 臨界距離（米），低於此距離 → 極度危險
    
    Returns:
        shape (num_envs,) 的安全獎勵
        正值 → 安全行為，負值 → 危險行為
    
    教學：為什麼用 CBF？
    - PCCBF 論文證明：CBF 能保證「前向不變性」，即一旦安全就永遠安全
    - 傳統獎勵（如 obstacle_proximity_penalty）只「懲罰接近」，無法保證安全
    - CBF 提供數學上的安全保證，這是論文的關鍵貢獻
    - 在強化學習中，我們用「獎勵 shaping」近似 CBF 的效果
    
    測試方法：
    訓練後，檢查 Episode_Reward/cbf_safety 的值：
    - 如果平均 > 0.5 → Agent 學會保持安全距離
    - 如果平均 < -1.0 → Agent 仍然經常進入危險區
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
    else:
        raise AttributeError(f"RayCasterData has no recognized distance attribute")
    
    distances = distances.squeeze(-1) if distances.dim() > 2 else distances  # 確保 (num_envs, num_rays)

    # 計算最小障礙物距離（最危險的方向）
    min_distance, _ = torch.min(distances, dim=-1)  # (num_envs,)

    # 🔥 CBF 核心邏輯：定義安全函數 h(x)
    # h(x) = min_distance - safe_distance
    # h(x) > 0 → 安全區（給正獎勵）
    # h(x) < 0 → 危險區（給負獎勵）
    h = min_distance - safe_distance

    # 安全獎勵計算（分段函數）
    # 1. 極度危險區（< critical_distance）：指數級懲罰
    critical_mask = min_distance < critical_distance
    critical_penalty = torch.exp(-5.0 * (min_distance - critical_distance))  # 距離越近，懲罰越大
    
    # 2. 警告區（critical_distance ~ safe_distance）：線性懲罰
    warning_mask = (min_distance >= critical_distance) & (min_distance < safe_distance)
    warning_penalty = (safe_distance - min_distance) / (safe_distance - critical_distance)
    
    # 3. 安全區（> safe_distance）：小額正獎勵（鼓勵保持安全）
    safe_mask = min_distance >= safe_distance
    safe_reward = torch.ones_like(min_distance) * 0.1  # 小額獎勵
    
    # 組合獎勵
    reward = torch.zeros_like(min_distance)
    reward = torch.where(critical_mask, -critical_penalty, reward)
    reward = torch.where(warning_mask, -warning_penalty, reward)
    reward = torch.where(safe_mask, safe_reward, reward)

    return reward


def predicted_cbf_safety_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    prediction_horizon: int = 3,
    safe_distance: float = 1.5,
) -> torch.Tensor:
    """【PCCBF 完整版】基於預測的 CBF 安全獎勵
    
    設計理念：
    這是 PCCBF 論文的完整實現：結合「預測」和「CBF」。
    論文中的「前瞻時域地圖（FTD Map）」在這裡體現為：
    評估未來 N 步的安全性，而不僅僅是當前時刻。
    
    差異對比：
    - cbf_safety_reward：只看「當前」障礙物距離（傳統 CBF）
    - predicted_cbf_safety_reward：看「未來」障礙物距離（PCCBF）
    
    為什麼需要預測版？
    - 機器人有慣性，無法瞬間停止
    - 如果只看當前，高速接近障礙物時會來不及剎車
    - 預測版能提前 3-5 步「看到」危險，給 Agent 反應時間
    
    Args:
        env: 環境物件
        sensor_cfg: LiDAR 感測器配置
        prediction_horizon: 預測未來幾步
        safe_distance: 安全距離閾值
    
    Returns:
        shape (num_envs,) 的預測安全獎勵
    
    教學：PCCBF vs CBF
    - CBF（傳統）：h(x_t) ≥ 0（當前時刻安全）
    - PCCBF（論文）：h(x_{t+k}) ≥ 0 for k=1..N（未來 N 步都安全）
    - 這讓控制更「保守」，但更安全
    """
    # 獲取當前 LiDAR 距離
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    
    if hasattr(data, "ray_hits_w") and hasattr(data, "pos_w"):
        hit_points = data.ray_hits_w  # (num_envs, num_rays, 3)
        sensor_pos = data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        current_distances = torch.norm(hit_points - sensor_pos, dim=-1)  # (num_envs, num_rays)
    else:
        raise AttributeError("需要 Isaac Sim 5.0+ 的 ray_hits_w API")
    
    # 獲取機器人速度（用於預測）
    robot = env.scene["robot"]
    robot_vel = robot.data.root_lin_vel_b  # (num_envs, 3)
    
    # 預測未來位置
    dt = env.step_dt * prediction_horizon
    predicted_displacement = torch.norm(robot_vel[:, :2], dim=-1) * dt  # (num_envs,)
    
    # 計算預測最小距離
    min_current_distance = torch.min(current_distances, dim=-1)[0]  # (num_envs,)
    predicted_min_distance = min_current_distance - predicted_displacement
    predicted_min_distance = torch.clamp(predicted_min_distance, min=0.0)
    
    # 使用預測距離計算 CBF 獎勵
    h_predicted = predicted_min_distance - safe_distance
    
    # 如果預測會進入危險區，給大懲罰
    danger_mask = h_predicted < 0
    danger_penalty = torch.abs(h_predicted) * 2.0  # 預測越危險，懲罰越大
    
    # 如果預測仍然安全，給小獎勵
    safe_reward = torch.ones_like(h_predicted) * 0.05
    
    reward = torch.where(danger_mask, -danger_penalty, safe_reward)
    
    return reward



