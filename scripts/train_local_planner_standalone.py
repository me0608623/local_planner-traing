#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
獨立的本地規劃器訓練腳本 - 避免 Pinocchio 依賴問題

這個腳本繞過 isaaclab_tasks 的自動載入機制，直接註冊和使用我們的環境。
"""

import argparse

from isaaclab.app import AppLauncher

# 解析命令列參數
parser = argparse.ArgumentParser(description="訓練 Nova Carter 本地規劃器（獨立版本）")
parser.add_argument("--num_envs", type=int, default=128, help="環境數量")
parser.add_argument("--headless", action="store_true", default=False, help="無頭模式")
parser.add_argument("--device", type=str, default="cuda:0", help="運算裝置")
parser.add_argument("--max_iterations", type=int, default=3000, help="最大訓練迭代次數")
parser.add_argument("--seed", type=int, default=42, help="隨機種子")

args_cli = parser.parse_args()

# 啟動 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必須在 AppLauncher 之後導入
import gymnasium as gym
import torch

# 直接導入我們的環境（避免載入 isaaclab_tasks 的其他模組）
from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg import (
    LocalPlannerEnvCfg,
    LocalPlannerEnvCfg_SIMPLE,
)
from isaaclab_tasks.manager_based.navigation.local_planner.agents.rsl_rl_ppo_cfg import (
    LocalPlannerPPORunnerCfg,
)
from isaaclab.envs import ManagerBasedRLEnv

# 手動註冊環境（避免自動載入）
gym.register(
    id="Isaac-Navigation-LocalPlanner-Carter-Standalone-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LocalPlannerEnvCfg},
)


def main():
    """主函數"""
    
    print("\n" + "=" * 80)
    print("Nova Carter 本地規劃器訓練（獨立版本 - 無 Pinocchio 依賴）")
    print("=" * 80 + "\n")
    
    # 建立環境配置
    env_cfg = LocalPlannerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # 建立環境
    print(f"建立環境... (num_envs={args_cli.num_envs})")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 建立 RL 演算法配置
    agent_cfg = LocalPlannerPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    agent_cfg.max_iterations = args_cli.max_iterations
    
    # 導入 RSL-RL
    from isaaclab_rl.rsl_rl.ppo import OnPolicyRunner
    
    # 建立 runner
    print(f"建立 PPO Runner...")
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    
    # 開始訓練
    print(f"\n開始訓練... (max_iterations={args_cli.max_iterations})")
    print("=" * 80 + "\n")
    
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    
    print("\n" + "=" * 80)
    print("訓練完成！")
    print("=" * 80 + "\n")
    
    # 關閉環境
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()





