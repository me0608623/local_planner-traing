# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL 強化學習訓練腳本

這個腳本用於訓練基於 RSL-RL 算法的強化學習 Agent。
RSL-RL 是蘇黎世聯邦理工學院（ETH Zurich）開發的強化學習框架，
專門用於機器人控制任務，支持 PPO 等 On-Policy 算法。

主要功能：
1. 啟動 Isaac Sim 模擬器
2. 創建並配置 RL 環境
3. 初始化 RSL-RL 訓練器
4. 執行訓練循環
5. 保存模型檢查點和訓練日誌
"""

"""第一階段：啟動 Isaac Sim 模擬器（必須在導入其他模組前完成）"""

# 導入命令行參數解析器
import argparse  # 用於解析命令行參數
import sys       # 用於修改系統參數和路徑

# 導入 Isaac Lab 的應用啟動器（用於初始化 Isaac Sim）
from isaaclab.app import AppLauncher

# 本地導入：命令行參數工具（必須在 AppLauncher 後導入）
import cli_args  # isort: skip


# ============================================================================
# 命令行參數設置
# ============================================================================
# 創建參數解析器，用於接收訓練相關的配置參數
parser = argparse.ArgumentParser(description="使用 RSL-RL 訓練強化學習 Agent")

# 視頻錄製相關參數
parser.add_argument("--video", action="store_true", default=False, 
                   help="是否在訓練過程中錄製視頻（用於可視化 Agent 行為）")
parser.add_argument("--video_length", type=int, default=200, 
                   help="每段錄製視頻的長度（單位：步數）")
parser.add_argument("--video_interval", type=int, default=2000, 
                   help="視頻錄製間隔（每隔多少步錄製一次）")

# 環境配置參數
parser.add_argument("--num_envs", type=int, default=None, 
                   help="並行模擬的環境數量（越多訓練越快，但需要更多GPU記憶體）")
parser.add_argument("--task", type=str, default=None, 
                   help="任務名稱（例如：Isaac-Navigation-LocalPlanner-PCCBF-Simple-v0）")

# Agent 配置參數
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", 
                   help="RL Agent 配置的入口點名稱（指定使用哪個 PPO 配置）")
parser.add_argument("--seed", type=int, default=None, 
                   help="隨機種子（用於結果可重現性）")
parser.add_argument("--max_iterations", type=int, default=None, 
                   help="訓練的最大迭代次數（例如：1000）")

# 分散式訓練參數
parser.add_argument("--distributed", action="store_true", default=False, 
                   help="是否使用多 GPU 或多節點進行分散式訓練")

# IO 描述符導出（用於調試和分析）
parser.add_argument("--export_io_descriptors", action="store_true", default=False, 
                   help="是否導出輸入輸出描述符（用於檢查觀測和動作空間）")

# 添加 RSL-RL 特定的命令行參數（例如：學習率、批次大小等）
cli_args.add_rsl_rl_args(parser)

# 添加 AppLauncher 的命令行參數（例如：--headless, --device 等）
AppLauncher.add_app_launcher_args(parser)

# 解析命令行參數：
# - args_cli: 本腳本定義的參數
# - hydra_args: Hydra 配置系統的參數（用於高級配置管理）
args_cli, hydra_args = parser.parse_known_args()

# ============================================================================
# 預處理參數
# ============================================================================
# 如果要錄製視頻，必須啟用相機（因為視頻需要渲染畫面）
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv，只保留 Hydra 參數
# 這是因為 Hydra 配置系統需要自己解析命令行參數
sys.argv = [sys.argv[0]] + hydra_args

# ============================================================================
# 啟動 Isaac Sim 模擬器
# ============================================================================
# 創建 AppLauncher 實例（初始化 Omniverse 應用）
app_launcher = AppLauncher(args_cli)
# 獲取模擬應用實例（這是 Isaac Sim 的核心對象）
simulation_app = app_launcher.app

"""
第二階段：檢查 RSL-RL 版本兼容性

Isaac Sim 啟動後，現在可以安全導入其他模組。
首先檢查 RSL-RL 版本是否滿足最低要求（分散式訓練需要特定版本）。
"""

# 導入版本檢查相關的模組
import importlib.metadata as metadata  # 用於查詢已安裝套件的版本
import platform                        # 用於檢測操作系統類型

from packaging import version          # 用於比較版本號

# ============================================================================
# RSL-RL 版本檢查（僅分散式訓練需要）
# ============================================================================
# 定義最低支援的 RSL-RL 版本
RSL_RL_VERSION = "2.3.1"

# 獲取當前安裝的 RSL-RL 版本
installed_version = metadata.version("rsl-rl-lib")

# 如果啟用分散式訓練且版本過舊，則提示升級
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    # 根據操作系統生成升級命令
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    
    # 顯示錯誤訊息並退出
    print(
        f"請安裝正確版本的 RSL-RL。\n當前版本：'{installed_version}'"
        f"，需要版本：'{RSL_RL_VERSION}'。\n要安裝正確版本，請執行：\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""
第三階段：導入所有必要的模組

版本檢查完成後，導入訓練所需的所有模組。
"""

# 標準庫導入
import gymnasium as gym          # Gymnasium（OpenAI Gym 的後繼者），用於 RL 環境接口
import os                         # 用於文件和目錄操作
import torch                      # PyTorch 深度學習框架
from datetime import datetime     # 用於生成時間戳（創建日誌目錄）

# Isaac Sim 相關導入
import omni                       # Omniverse 核心模組（用於日誌輸出）

# RSL-RL 核心導入
from rsl_rl.runners import OnPolicyRunner  # RSL-RL 的訓練執行器（負責訓練循環）

# Isaac Lab 環境相關導入
from isaaclab.envs import (
    DirectMARLEnv,              # 直接式多智能體 RL 環境
    DirectMARLEnvCfg,           # 直接式多智能體環境配置
    DirectRLEnvCfg,             # 直接式單智能體環境配置
    ManagerBasedRLEnvCfg,       # 管理器式環境配置（我們的導航任務使用這個）
    multi_agent_to_single_agent, # 多智能體轉單智能體的轉換器
)

# Isaac Lab 工具函數導入
from isaaclab.utils.dict import print_dict  # 用於美化打印字典
from isaaclab.utils.io import dump_pickle, dump_yaml  # 用於保存配置文件

# Isaac Lab RL 包裝器導入
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Isaac Lab 任務註冊（導入後自動註冊所有環境，包括我們的 PCCBF 環境）
import isaaclab_tasks  # noqa: F401

# Isaac Lab 工具函數
from isaaclab_tasks.utils import get_checkpoint_path  # 用於查找模型檢查點
from isaaclab_tasks.utils.hydra import hydra_task_config  # Hydra 配置裝飾器

# PLACEHOLDER: Extension template (do not remove this comment)

# ============================================================================
# PyTorch 性能優化設置
# ============================================================================
# 啟用 TensorFloat-32（TF32）以加速訓練（在 Ampere GPU 上，如 RTX 3090/4090）
torch.backends.cuda.matmul.allow_tf32 = True     # 矩陣乘法使用 TF32
torch.backends.cudnn.allow_tf32 = True           # cuDNN 卷積使用 TF32

# 關閉確定性模式以提升性能（訓練結果可能會有微小差異）
torch.backends.cudnn.deterministic = False

# 關閉 cuDNN benchmark（因為我們的輸入大小固定，不需要動態選擇算法）
torch.backends.cudnn.benchmark = False


# ============================================================================
# 主訓練函數
# ============================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """
    使用 RSL-RL Agent 執行強化學習訓練
    
    這是整個訓練腳本的核心函數，負責：
    1. 配置環境和 Agent 參數
    2. 創建 RL 環境
    3. 初始化訓練器
    4. 執行訓練循環
    5. 保存模型和日誌
    
    Args:
        env_cfg: 環境配置對象（包含場景、觀測、動作、獎勵等所有設置）
        agent_cfg: Agent 配置對象（包含 PPO 超參數、網絡架構等）
        
    注意：
        - @hydra_task_config 裝飾器會自動從註冊的任務中載入配置
        - 例如：--task Isaac-Navigation-LocalPlanner-PCCBF-Simple-v0
              會載入對應的 env_cfg 和 agent_cfg
    """
    
    # ========================================================================
    # 步驟 1：配置參數覆蓋（命令行參數優先於配置文件）
    # ========================================================================
    # 用命令行參數更新 Agent 配置（例如：學習率、批次大小等）
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # 如果命令行指定了環境數量，則覆蓋配置文件中的值
    # 三元運算符：如果 args_cli.num_envs 不是 None，使用它；否則使用配置文件的值
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # 如果命令行指定了最大迭代次數，則覆蓋配置文件中的值
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # ========================================================================
    # 步驟 2：設置隨機種子（確保結果可重現）
    # ========================================================================
    # 注意：某些隨機化發生在環境初始化時，所以在這裡設置種子
    env_cfg.seed = agent_cfg.seed
    
    # 設置模擬設備（GPU 或 CPU）
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ========================================================================
    # 步驟 3：多 GPU 訓練配置（如果啟用分散式訓練）
    # ========================================================================
    if args_cli.distributed:
        # 為每個 GPU 進程分配獨立的 CUDA 設備
        # local_rank 是當前進程在當前節點上的 GPU 索引（0, 1, 2, ...）
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 為每個進程設置不同的隨機種子，以增加數據多樣性
        # 例如：seed=42, local_rank=0 → seed=42; local_rank=1 → seed=43
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # ========================================================================
    # 步驟 4：設置日誌目錄
    # ========================================================================
    # 構建日誌根目錄：logs/rsl_rl/{實驗名稱}
    # 例如：logs/rsl_rl/local_planner_carter/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)  # 轉換為絕對路徑
    print(f"[INFO] 實驗日誌目錄：{log_root_path}")
    
    # 為本次訓練創建帶時間戳的子目錄：{年-月-日}_{時-分-秒}
    # 例如：2025-10-25_14-30-45
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 重要：Ray Tune 工作流程使用下面這行日誌提取實驗名稱，不要修改格式
    # （參考 PR #2346, comment-2819298849）
    print(f"命令行請求的實驗名稱：{log_dir}")
    
    # 如果指定了運行名稱，則附加到目錄名稱後
    # 例如：2025-10-25_14-30-45_pccbf_test
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    
    # 組合成完整的日誌目錄路徑
    # 例如：logs/rsl_rl/local_planner_carter/2025-10-25_14-30-45/
    log_dir = os.path.join(log_root_path, log_dir)

    # ========================================================================
    # 步驟 5：設置 IO 描述符導出（如果請求）
    # ========================================================================
    # IO 描述符用於調試，可以查看觀測和動作的詳細信息
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        # 只有管理器式環境支持 IO 描述符導出
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        # 直接式環境不支持，發出警告
        omni.log.warn(
            "IO 描述符僅支持管理器式 RL 環境。不會導出 IO 描述符。"
        )

    # ========================================================================
    # 步驟 6：創建 Isaac 環境
    # ========================================================================
    # 使用 Gymnasium 的 make 函數創建環境
    # - args_cli.task: 任務名稱（例如：Isaac-Navigation-LocalPlanner-PCCBF-Simple-v0）
    # - cfg: 環境配置對象
    # - render_mode: 如果錄製視頻則設為 "rgb_array"（渲染 RGB 圖像）
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ========================================================================
    # 步驟 7：多智能體轉單智能體（如果需要）
    # ========================================================================
    # 如果是多智能體環境，但 RL 算法只支持單智能體，則進行轉換
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ========================================================================
    # 步驟 8：獲取恢復訓練的檢查點路徑（如果需要）
    # ========================================================================
    # 在創建新的 log_dir 之前保存恢復路徑
    # 兩種情況需要載入檢查點：
    # 1. agent_cfg.resume = True：從中斷的訓練繼續
    # 2. 使用 Distillation 算法：需要載入教師模型
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ========================================================================
    # 步驟 9：包裝環境以支持視頻錄製（如果請求）
    # ========================================================================
    if args_cli.video:
        # 視頻錄製配置
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),  # 視頻保存目錄
            "step_trigger": lambda step: step % args_cli.video_interval == 0,  # 每隔多少步錄製一次
            "video_length": args_cli.video_length,  # 每段視頻的長度
            "disable_logger": True,  # 禁用額外的日誌輸出
        }
        print("[INFO] 訓練過程中將錄製視頻。")
        print_dict(video_kwargs, nesting=4)  # 美化打印視頻配置
        # 使用 Gymnasium 的 RecordVideo 包裝器包裝環境
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ========================================================================
    # 步驟 10：包裝環境以適配 RSL-RL
    # ========================================================================
    # RslRlVecEnvWrapper 將 Isaac Lab 環境轉換為 RSL-RL 期望的接口
    # - clip_actions: 是否將動作裁剪到 [-1, 1] 範圍
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ========================================================================
    # 步驟 11：創建 RSL-RL 訓練器
    # ========================================================================
    # OnPolicyRunner 是 RSL-RL 的核心訓練執行器，負責：
    # - 管理 PPO 算法的訓練循環
    # - 收集經驗（rollout）
    # - 更新策略網絡
    # - 記錄訓練指標（TensorBoard）
    # - 保存模型檢查點
    runner = OnPolicyRunner(
        env,                      # 包裝後的環境
        agent_cfg.to_dict(),      # Agent 配置（轉換為字典）
        log_dir=log_dir,          # 日誌目錄
        device=agent_cfg.device   # 訓練設備（cuda:0 或 cpu）
    )
    
    # ========================================================================
    # 步驟 12：記錄 Git 狀態（用於實驗追蹤）
    # ========================================================================
    # 將當前 Git 倉庫的狀態（commit hash、branch 等）記錄到日誌
    # 這對於重現實驗非常重要
    runner.add_git_repo_to_log(__file__)
    
    # ========================================================================
    # 步驟 13：載入檢查點（如果需要）
    # ========================================================================
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: 從檢查點載入模型：{resume_path}")
        # 載入之前訓練的模型權重和優化器狀態
        runner.load(resume_path)

    # ========================================================================
    # 步驟 14：保存配置文件到日誌目錄
    # ========================================================================
    # 保存環境和 Agent 配置，方便日後查看和重現實驗
    # 同時保存為 YAML（人類可讀）和 Pickle（Python 對象）格式
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)      # 環境配置 YAML
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)  # Agent 配置 YAML
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)     # 環境配置 Pickle
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg) # Agent 配置 Pickle

    # ========================================================================
    # 步驟 15：開始訓練！🚀
    # ========================================================================
    # 這是整個腳本的核心：執行訓練循環
    # - num_learning_iterations: 訓練的總迭代次數（例如：1000）
    # - init_at_random_ep_len: 是否在隨機 episode 長度處初始化
    #   （True 可以增加數據多樣性，避免所有環境同步重置）
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ========================================================================
    # 步驟 16：清理資源
    # ========================================================================
    # 訓練完成後關閉環境（釋放 GPU 記憶體和其他資源）
    env.close()


# ============================================================================
# 腳本入口點
# ============================================================================
if __name__ == "__main__":
    # 執行主訓練函數
    # @hydra_task_config 裝飾器會自動處理配置載入
    main()
    
    # 關閉 Isaac Sim 應用
    # 這會釋放所有 Omniverse 相關的資源
    simulation_app.close()
