# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 PPO 演算法配置 - 本地規劃器任務"""

# Stable-Baselines3 PPO 配置字典 - Nova Carter 本地規劃器
LocalPlannerSB3PPORunnerCfg = {
    # 基本訓練配置
    "seed": 42,
    "device": "cuda:0",  # 明確指定使用 GPU
    "policy": "MlpPolicy",  # SB3 策略類型 (必需)
    "n_timesteps": 1_000_000,  # 總訓練步數
    
    # PPO 演算法超參數
    "learning_rate": 3e-4,
    "n_steps": 2048,  # 每次更新收集的步數
    "batch_size": 128,  # 小批次大小
    "n_epochs": 10,  # 每次更新的訓練輪數
    "gamma": 0.99,  # 折扣因子
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # PPO 裁剪範圍
    "ent_coef": 0.01,  # 熵係數
    "vf_coef": 0.5,  # 價值函數係數
    "max_grad_norm": 0.5,  # 梯度裁剪
    
    # 網路架構配置
    "policy_kwargs": {
        "net_arch": [256, 256, 128],  # actor 和 critic 共享網路架構
        "activation_fn": "nn.ELU",  # 激活函數 (SB3 格式)
        "ortho_init": False,  # 正交初始化
    },
    
    # 其他 PPO 配置
    "use_sde": False,  # 是否使用狀態相關探索
    "sde_sample_freq": -1,  # SDE 採樣頻率
    "target_kl": None,  # 目標 KL 散度
}
