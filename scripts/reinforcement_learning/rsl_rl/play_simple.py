#!/usr/bin/env python3
"""簡化的模型測試腳本 - 跳過ONNX導出，只看GUI效果"""

import argparse
from isaaclab.app import AppLauncher

# 解析參數
parser = argparse.ArgumentParser(description="測試訓練好的模型（簡化版）")
parser.add_argument("--task", type=str, required=True, help="環境名稱")
parser.add_argument("--num_envs", type=int, default=1, help="環境數量")
parser.add_argument("--checkpoint", type=str, required=True, help="模型路徑")
# 不要自己定義--headless，AppLauncher會自動加入

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 啟動Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 導入其他模組
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 載入環境和agent配置
from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg_debug import (
    LocalPlannerEnvCfg_DEBUG
)
from isaaclab_tasks.manager_based.navigation.local_planner.agents.rsl_rl_ppo_cfg import (
    LocalPlannerPPORunnerCfg
)

# 創建環境配置
env_cfg = LocalPlannerEnvCfg_DEBUG()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = "cuda:0"

# 創建環境
print(f"[INFO] 創建環境：{args.task}")
env = gym.make(args.task, cfg=env_cfg)

# 包裝環境（RSL-RL需要）
env = RslRlVecEnvWrapper(env)

# 創建runner（只用於載入模型）
agent_cfg = LocalPlannerPPORunnerCfg()
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device='cuda:0')

# 載入checkpoint
print(f"[INFO] 載入模型：{args.checkpoint}")
runner.load(args.checkpoint)

# 獲取推理用的policy
policy = runner.get_inference_policy(device=env.unwrapped.device)

# 測試模型
print(f"[INFO] 開始測試...（按Ctrl+C停止）")
print("=" * 80)

obs, _ = env.get_observations()
episode_count = 0
success_count = 0

try:
    while simulation_app.is_running():
        # 模型推理（使用RSL-RL標準方式）
        with torch.inference_mode():
            actions = policy(obs)
        
        # 執行動作（RslRlVecEnvWrapper返回4個值）
        obs, _, _, extras = env.step(actions)
        
        # 統計（簡化 - 只在終端顯示，主要看GUI）
        # 每100步顯示一次進度
        if hasattr(env, 'episode_length_buf'):
            # 檢查是否有episode結束
            episode_lengths = env.unwrapped.episode_length_buf
            if (episode_lengths == 0).any():  # 剛重置的環境
                episode_count += 1
                
                # 每5個episodes顯示一次
                if episode_count % 5 == 0:
                    print(f"[INFO] 已完成 {episode_count} 個episodes（請觀察GUI中的導航行為）")

except KeyboardInterrupt:
    print("\n" + "=" * 80)
    print(f"[INFO] 測試結束")
    print(f"總Episodes: {episode_count}")
    print(f"成功: {success_count}")
    if episode_count > 0:
        print(f"成功率: {(success_count/episode_count)*100:.2f}%")
    print("=" * 80)

# 關閉
env.close()
simulation_app.close()

