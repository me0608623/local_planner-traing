#!/bin/bash

# ============================================================
# 🚀 v4 訓練啟動腳本
# ============================================================
# 版本: v4 (方案 A - 平衡正負獎勵)
# 日期: 2025-10-30
# 目標: Progress 回正 + 懲罰降低
# ============================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║          🤖 Nova Carter Navigation v4 訓練                 ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 設定 WandB API Key
export WANDB_API_KEY="015e82155dc7cecb79368e0415eaf09362c260ef"
echo "✅ WandB API Key 已設定"
echo ""

# 顯示配置
echo "📊 訓練配置："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  任務名稱      │ Isaac-Navigation-LocalPlanner-Min-v0"
echo "  RL 算法       │ PPO (Proximal Policy Optimization)"
echo "  框架          │ RSL-RL"
echo "  ────────────────────────────────────────────────────────"
echo "  並行環境數    │ 24"
echo "  訓練迭代數    │ 10000"
echo "  每步時長      │ 0.04s (物理 dt: 0.01s)"
echo "  Episode 長度  │ 60s (1500 步)"
echo "  ────────────────────────────────────────────────────────"
echo "  學習率        │ 3e-4"
echo "  熵係數        │ 0.001"
echo "  Clip 範圍     │ 0.1"
echo "  ────────────────────────────────────────────────────────"
echo "  記錄工具      │ WandB"
echo "  專案名稱      │ nova-carter-navigation"
echo "  設備          │ cuda:0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🎯 v4 核心調整（vs v3）："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  懲罰項大幅降低（避免壓制探索）："
echo "    • standstill_penalty: 4.0 → 1.0   (↓ 75%)"
echo "    • anti_idle:          2.0 → 0.5   (↓ 75%)"
echo "    • spin_penalty:       0.5 → 0.1   (↓ 80%)"
echo "    • time_penalty:       0.01 → 0.005 (↓ 50%)"
echo ""
echo "  正向獎勵保持強力："
echo "    • progress_to_goal:   60.0 (保持)"
echo "    • near_goal_shaping:  20.0 (保持, radius 3.0m)"
echo "    • heading_alignment:  1.0  (保持, 條件式)"
echo "    • reached_goal:       200.0 (保持)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "⏱️  預估時間："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  總訓練時間    │ 約 3.5 小時"
echo "  1000 iter     │ 約 20 分鐘"
echo "  5000 iter     │ 約 1.75 小時"
echo "  10000 iter    │ 約 3.5 小時"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🎯 v4 成功標準（10000 iter）："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ⭐⭐⭐ Progress > 0.05        (v3: -0.0104)"
echo "  ⭐⭐⭐ Standstill > -0.3      (v3: -1.16)"
echo "  ⭐⭐⭐ Position Error < 3.0m (v3: 3.84m)"
echo "  ⭐⭐  Success Rate > 0%      (v3: 0%)"
echo "  ⭐⭐  Near Goal > 0.1        (v3: 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📈 監控方式："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. WandB 即時監控（推薦）"
echo "     → https://wandb.ai/"
echo "     → 專案：nova-carter-navigation"
echo ""
echo "  2. 終端機即時日誌"
echo "     → 另開終端執行："
echo "     → tail -f /home/aa/IsaacLab/training_v4.log"
echo ""
echo "  3. 關鍵檢查點"
echo "     → T+30min (1000 iter):  Progress 是否回正？"
echo "     → T+1.5h  (3000 iter):  Position Error 改善？"
echo "     → T+2.5h  (5000 iter):  Success Rate 出現？"
echo "     → T+3.5h  (10000 iter): 完整評估"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🚀 正在啟動訓練..."
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# 啟動訓練
cd /home/aa/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Min-v0 \
    --num_envs 24 \
    --max_iterations 10000 \
    --headless

