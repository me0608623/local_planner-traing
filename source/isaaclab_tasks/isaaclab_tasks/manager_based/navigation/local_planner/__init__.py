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
from .local_planner_env_cfg_gui_fixed import (
    LocalPlannerEnvCfg_GUI_FIXED,
    LocalPlannerEnvCfg_GUI_SIMPLE,
    LocalPlannerEnvCfg_DIAGNOSTIC
)
# 🔥 PCCBF-MPC 啟發版本（基於 2025 年論文）
from .local_planner_env_cfg_pccbf import (
    LocalPlannerEnvCfg_PCCBF_EASY,
    LocalPlannerEnvCfg_PCCBF_MEDIUM,
    LocalPlannerEnvCfg_PCCBF_HARD,
)
# 🔥 PCCBF 簡化版（推薦先用這個）
from .local_planner_env_cfg_pccbf_simple import (
    LocalPlannerEnvCfg_PCCBF_SIMPLE,
)
# 🔬 DEBUG 版本（診斷用）
from .local_planner_env_cfg_debug import (
    LocalPlannerEnvCfg_DEBUG,
)
# ✅ Simple v2（基於DEBUG成功經驗的優化版）
from .local_planner_env_cfg_simple_v2 import (
    LocalPlannerEnvCfg_SIMPLE_V2_STAGE1,
    LocalPlannerEnvCfg_SIMPLE_V2_STAGE1_5,
    LocalPlannerEnvCfg_SIMPLE_V2_STAGE2,
    LocalPlannerEnvCfg_SIMPLE_V2_STAGE3,
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

# GUI 模式專用環境 - 基於GUI vs Headless重要發現
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gui_fixed:LocalPlannerEnvCfg_GUI_FIXED",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-GUI-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gui_fixed:LocalPlannerEnvCfg_GUI_SIMPLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 診斷專用環境
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-Diagnostic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_gui_fixed:LocalPlannerEnvCfg_DIAGNOSTIC",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

##
# 🔥 PCCBF-MPC 啟發版本（基於 2025 年論文）
# 論文：Point Cloud-Based Control Barrier Functions for MPC
##

# 🎯 PCCBF 簡化版（推薦！穩定性最高）
gym.register(
    id="Isaac-Navigation-LocalPlanner-PCCBF-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_pccbf_simple:LocalPlannerEnvCfg_PCCBF_SIMPLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🎓 課程學習 - 階段 1：簡單版（帶預測觀測）
gym.register(
    id="Isaac-Navigation-LocalPlanner-PCCBF-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_pccbf:LocalPlannerEnvCfg_PCCBF_EASY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🎓 課程學習 - 階段 2：中等版（EASY 成功後進階）
gym.register(
    id="Isaac-Navigation-LocalPlanner-PCCBF-Medium-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_pccbf:LocalPlannerEnvCfg_PCCBF_MEDIUM",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🎓 課程學習 - 階段 3：困難版（MEDIUM 成功後進階）
gym.register(
    id="Isaac-Navigation-LocalPlanner-PCCBF-Hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_pccbf:LocalPlannerEnvCfg_PCCBF_HARD",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# 🔬 DEBUG 版本（診斷用 - 極簡配置）
gym.register(
    id="Isaac-Navigation-LocalPlanner-DEBUG-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_debug:LocalPlannerEnvCfg_DEBUG",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

# ✅ Simple v2（基於DEBUG成功，逐步增加難度）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Simple-v2-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_simple_v2:LocalPlannerEnvCfg_SIMPLE_V2_STAGE1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Simple-v2-Stage1.5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_simple_v2:LocalPlannerEnvCfg_SIMPLE_V2_STAGE1_5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Simple-v2-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_simple_v2:LocalPlannerEnvCfg_SIMPLE_V2_STAGE2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-LocalPlanner-Simple-v2-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.local_planner_env_cfg_simple_v2:LocalPlannerEnvCfg_SIMPLE_V2_STAGE3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}.sb3_ppo_cfg:LocalPlannerSB3PPORunnerCfg",
    },
)