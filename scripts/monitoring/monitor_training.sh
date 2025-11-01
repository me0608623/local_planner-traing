#!/bin/bash

# ============================================================
# 📊 v4 訓練即時監控腳本
# ============================================================

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║           📊 v4 訓練即時監控面板                           ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 檢查訓練進程
TRAIN_PID=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ 訓練進程未運行"
    echo ""
    echo "啟動訓練："
    echo "  cd /home/aa/IsaacLab && ./start_v4_training_gui.sh"
    exit 1
fi

echo "✅ 訓練進程運行中（PID: $TRAIN_PID）"
echo ""

# 檢查 Isaac Sim 進程
ISAAC_COUNT=$(ps aux | grep -E "kit|omni" | grep -v grep | wc -l)
echo "✅ Isaac Sim 進程數：$ISAAC_COUNT"
echo ""

# 檢查 GPU 使用
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 GPU 狀態："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n  使用率: %s%% | 記憶體: %s/%s MB\n\n", $1, $2, $3, $4, $5}'
echo ""

# 查找最新日誌目錄
LATEST_LOG_DIR=$(ls -dt logs/rsl_rl/local_planner_carter/2025-* 2>/dev/null | head -1)

if [ -z "$LATEST_LOG_DIR" ]; then
    echo "⏳ 訓練日誌尚未生成（場景初始化中...）"
    echo ""
    echo "預計等待時間："
    echo "  • 場景初始化：1-3 分鐘"
    echo "  • GUI 視窗出現：2-5 分鐘"
    echo "  • 開始訓練：3-6 分鐘"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💡 提示："
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  1. Isaac Sim GUI 視窗會自動彈出"
    echo "  2. 請檢查其他桌面/工作區"
    echo "  3. 重新執行此腳本查看更新："
    echo "     ./monitor_training.sh"
    echo ""
else
    echo "📁 日誌目錄：$LATEST_LOG_DIR"
    echo ""
    
    # 查找 WandB 日誌
    WANDB_LOG=$(find "$LATEST_LOG_DIR" -name "*.log" 2>/dev/null | head -1)
    
    if [ -z "$WANDB_LOG" ]; then
        echo "⏳ WandB 日誌尚未生成"
        echo ""
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📈 最新訓練數據："
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 提取最新迭代
        LATEST_ITER=$(grep -o "Learning iteration [0-9]*/10000" "$WANDB_LOG" 2>/dev/null | tail -1)
        if [ ! -z "$LATEST_ITER" ]; then
            echo "  $LATEST_ITER"
            echo ""
        fi
        
        # 提取 Mean Reward
        MEAN_REWARD=$(grep "Mean reward:" "$WANDB_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
        if [ ! -z "$MEAN_REWARD" ]; then
            echo "  Mean Reward: $MEAN_REWARD"
        fi
        
        # 提取 Progress
        PROGRESS=$(grep "progress_to_goal" "$WANDB_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
        if [ ! -z "$PROGRESS" ]; then
            echo "  Progress: $PROGRESS  ← 關鍵！應該 > 0"
        fi
        
        # 提取 Standstill
        STANDSTILL=$(grep "standstill" "$WANDB_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
        if [ ! -z "$STANDSTILL" ]; then
            echo "  Standstill: $STANDSTILL  ← 應該 > -0.5"
        fi
        
        # 提取 Position Error
        POS_ERROR=$(grep "position_error" "$WANDB_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
        if [ ! -z "$POS_ERROR" ]; then
            echo "  Position Error: $POS_ERROR  ← 應該 < 3.0m"
        fi
        
        echo ""
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 監控連結："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WandB 即時監控："
echo "  → https://wandb.ai/"
echo "  → 專案：nova-carter-navigation"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💻 快速指令："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • 持續監控：watch -n 5 ./monitor_training.sh"
echo "  • 停止訓練：pkill -f train.py"
echo "  • 查看詳細日誌：tail -f $LATEST_LOG_DIR/*.log"
echo ""

