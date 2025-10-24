#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
測試本地規劃器環境 - 修復 Pinocchio/Assimp 衝突版本

在導入任何模組前，設置 LD_LIBRARY_PATH 使用 Isaac Sim 的 Assimp
"""

# ⚠️ 重要：必須在任何導入前設置環境變數
import os
import sys

# 方案：使用 Isaac Sim 的 Assimp 庫
isaac_sim_lib = os.path.expanduser("~/IsaacLab/_isaac_sim/kit/lib")
current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

if isaac_sim_lib not in current_ld_path:
    os.environ["LD_LIBRARY_PATH"] = f"{isaac_sim_lib}:{current_ld_path}"
    print(f"✓ 已設置 LD_LIBRARY_PATH 使用 Isaac Sim 的庫")
    print(f"  {isaac_sim_lib}")

# 現在才導入其他模組
import argparse
from isaaclab.app import AppLauncher

# 解析命令列參數
parser = argparse.ArgumentParser(description="測試 Nova Carter 本地規劃器環境（修復版）")
parser.add_argument("--num_envs", type=int, default=1, help="環境數量")
parser.add_argument("--task", type=str, default="Isaac-Navigation-LocalPlanner-Carter-Simple-v0", help="任務名稱")
parser.add_argument("--headless", action="store_true", default=False, help="無頭模式運行")
parser.add_argument("--device", type=str, default="cuda:0", help="運算裝置")

args_cli = parser.parse_args()

# 啟動 Isaac Sim 應用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入必要的套件（必須在 AppLauncher 之後）
import gymnasium as gym
import torch

# 只導入我們的環境，避免載入使用 Pinocchio 的模組
import isaaclab_tasks.manager_based.navigation.local_planner  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """主函數：測試環境"""
    
    print("\n" + "=" * 80)
    print(f"測試環境: {args_cli.task}")
    print("已啟用 Pinocchio/Assimp 衝突修復")
    print("=" * 80 + "\n")
    
    # 解析環境配置
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    
    # 建立環境
    print(f"建立環境... (num_envs={args_cli.num_envs})")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print(f"✓ 環境建立成功！")
    print(f"  - 觀察空間: {env.observation_space}")
    print(f"  - 動作空間: {env.action_space}")
    print(f"  - 回合長度: {env.max_episode_length}")
    
    # 重置環境
    print("\n重置環境...")
    obs, info = env.reset()
    print(f"✓ 重置成功！")
    print(f"  - 觀察形狀: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
    
    # 運行隨機動作測試
    print("\n運行 100 步隨機動作測試...")
    for step in range(100):
        # 隨機動作
        action = 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0
        
        # 執行步驟
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每 20 步印出資訊
        if (step + 1) % 20 == 0:
            done = terminated | truncated
            print(f"  步驟 {step + 1}/100:")
            print(f"    - 平均獎勵: {reward.mean().item():.3f}")
            print(f"    - 已完成環境數: {done.sum().item()}/{args_cli.num_envs}")
    
    print("\n✓ 測試完成！環境運作正常。")
    print("=" * 80 + "\n")
    
    # 關閉環境
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()





