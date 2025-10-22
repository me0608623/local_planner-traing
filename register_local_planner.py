#!/usr/bin/env python3
"""
手動註冊 Nova Carter 本地規劃器環境
繞過自動導入系統，直接註冊環境到 Gymnasium
"""

import sys
import os

# 確保正確的模組路徑
sys.path.insert(0, 'source')

import gymnasium as gym

# 手動註冊環境
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg:LocalPlannerEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.navigation.local_planner.agents.rsl_rl_ppo_cfg:LocalPlannerPPORunnerCfg",
    },
)

print("✅ Nova Carter 本地規劃器環境已手動註冊")

# 驗證註冊
try:
    spec = gym.spec("Isaac-Navigation-LocalPlanner-Carter-v0")
    print(f"✅ 環境註冊驗證成功")
    print(f"   Entry point: {spec.entry_point}")
    print(f"   Env config: {spec.kwargs['env_cfg_entry_point']}")
    print(f"   Agent config: {spec.kwargs['rsl_rl_cfg_entry_point']}")
except Exception as e:
    print(f"❌ 環境註冊驗證失敗: {e}")

