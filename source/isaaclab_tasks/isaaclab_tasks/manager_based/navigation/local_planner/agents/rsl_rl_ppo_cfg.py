# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO 演算法配置 - 本地規劃器任務"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LocalPlannerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO Runner 配置 - Nova Carter 本地規劃器
    
    採用與其他 Isaac Lab 任務一致的簡潔配置風格。
    RSL-RL 會自動處理觀測分組，無需手動配置 obs_groups。
    """

    # 基本配置
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "local_planner_carter"
    run_name: str = ""  # 運行名稱，會附加在日誌目錄後
    empirical_normalization: bool = False
    
    # 觀測組配置 - 設為 None 讓 RSL-RL 自動推斷
    obs_groups: dict | None = None
    
    # 網路架構配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    
    # PPO 演算法超參數
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=3e-4,  # 🔧 從1e-3降到3e-4（0.0003），提升穩定性
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

