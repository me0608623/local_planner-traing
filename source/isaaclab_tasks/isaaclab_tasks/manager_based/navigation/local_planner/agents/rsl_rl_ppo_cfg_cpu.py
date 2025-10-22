# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO 演算法配置 - 本地規劃器任務 (CPU模式)"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LocalPlannerPPORunnerCfg_CPU(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO Runner 配置 - Nova Carter 本地規劃器 (CPU模式)
    
    專門配置用於修復PhysX GPU/CPU張量設備不匹配問題：
    - 錯誤：expected device 0, received device -1
    - 解決：強制所有訓練組件使用 CPU
    """

    # 基本配置 - CPU模式
    seed: int = 42
    device: str = "cpu"  # 🔧 核心修復：設為CPU與環境一致
    num_steps_per_env: int = 24
    max_iterations: int = 1000  # 🔧 CPU模式下減少迭代次數
    save_interval: int = 50     # 🔧 更頻繁保存（因為訓練較慢）
    experiment_name: str = "local_planner_carter_cpu"
    run_name: str = ""
    empirical_normalization: bool = False
    
    # 觀測組配置 - 設為 None 讓 RSL-RL 自動推斷
    obs_groups: dict | None = None
    
    # 網路架構配置 - CPU優化
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],  # 🔧 較小網絡以適應CPU性能
        critic_hidden_dims=[128, 128], # 🔧 較小網絡以適應CPU性能
        activation="elu",
    )
    
    # PPO 演算法超參數 - CPU優化
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=3,    # 🔧 減少學習輪數以加快CPU訓練
        num_mini_batches=2,       # 🔧 減少批次數量
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

