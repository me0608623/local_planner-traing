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
from .local_planner_env_cfg_cpu import (
    LocalPlannerEnvCfg_CPU,
    LocalPlannerEnvCfg_CPU_SIMPLE,
    LocalPlannerEnvCfg_GPU_FIXED
)
from .local_planner_env_cfg_gpu_optimized import (
    LocalPlannerEnvCfg_GPU_OPTIMIZED,
    LocalPlannerEnvCfg_GPU_OPTIMIZED_SIMPLE
)
from .local_planner_env_cfg_gpu_optimized_fixed import (
    LocalPlannerEnvCfg_GPU_OPTIMIZED_FIXED,
    LocalPlannerEnvCfg_GPU_OPTIMIZED_SIMPLE_FIXED
)
from .local_planner_env_cfg_isaac_sim_5_fixed import (
    LocalPlannerEnvCfg_ISAAC_SIM_5_FIXED,
    LocalPlannerEnvCfg_ISAAC_SIM_5_SIMPLE
)

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

# 🔧 CPU 模式版本（修復 GPU/CPU 張量不匹配）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-CPU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_cpu:LocalPlannerEnvCfg_CPU",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_cpu:LocalPlannerPPORunnerCfg_CPU",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-CPU-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_cpu:LocalPlannerEnvCfg_CPU_SIMPLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_cpu:LocalPlannerPPORunnerCfg_CPU",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🔧 GPU修復實驗版本
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_cpu:LocalPlannerEnvCfg_GPU_FIXED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🔧 GPU深度優化版本（路線A：全程GPU）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GPU-Optimized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gpu_optimized:LocalPlannerEnvCfg_GPU_OPTIMIZED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GPU-Optimized-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gpu_optimized:LocalPlannerEnvCfg_GPU_OPTIMIZED_SIMPLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🔧 GPU深度優化修復版本（不依賴omni.isaac.core）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gpu_optimized_fixed:LocalPlannerEnvCfg_GPU_OPTIMIZED_FIXED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gpu_optimized_fixed:LocalPlannerEnvCfg_GPU_OPTIMIZED_SIMPLE_FIXED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🎯 Isaac Sim 5.0 完全兼容版本（推薦）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_isaac_sim_5_fixed:LocalPlannerEnvCfg_ISAAC_SIM_5_FIXED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_isaac_sim_5_fixed:LocalPlannerEnvCfg_ISAAC_SIM_5_SIMPLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)