# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Nova Carter 本地規劃器環境 - 動態避障導航任務
"""

import gymnasium as gym

from . import agents
from .local_planner_env_cfg import LocalPlannerEnvCfg, LocalPlannerEnvCfg_SIMPLE
from .local_planner_env_cfg_min import LocalPlannerEnvCfgMin

##
# Register Gym environments
##

# 標準GPU版本
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg:LocalPlannerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg:LocalPlannerEnvCfg_SIMPLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# ✅ 最小環境版本（推薦：穩定訓練）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Min-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_min:LocalPlannerEnvCfgMin",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)
