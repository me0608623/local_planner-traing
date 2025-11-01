#!/bin/bash

# ============================================================
# 🚀 Nova Carter 訓練啟動器
# ============================================================
# 統一的訓練啟動入口
# ============================================================

clear

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║         🤖 Nova Carter Navigation 訓練啟動器              ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "請選擇訓練模式："
echo ""
echo "  1️⃣  無頭模式（最快，推薦）"
echo "     - 無 GUI 視窗"
echo "     - 訓練速度最快"
echo "     - 使用 WandB 監控"
echo ""
echo "  2️⃣  GUI 模式（可視化）"
echo "     - 顯示 Isaac Sim 視窗"
echo "     - 可看到 24 個機器人訓練"
echo "     - 訓練速度較慢"
echo ""
echo "  3️⃣  監控現有訓練"
echo "     - 查看訓練狀態"
echo "     - 不啟動新訓練"
echo ""
echo "  0️⃣  退出"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "請輸入選項 [1/2/3/0]: " choice
echo ""

case $choice in
    1)
        echo "🚀 啟動無頭模式訓練..."
        echo ""
        ./scripts/training/start_v4_training.sh
        ;;
    2)
        echo "🎮 啟動 GUI 模式訓練..."
        echo ""
        ./scripts/training/啟動24環境訓練.sh
        ;;
    3)
        echo "📊 監控現有訓練..."
        echo ""
        ./scripts/monitoring/monitor_training.sh
        ;;
    0)
        echo "👋 再見！"
        exit 0
        ;;
    *)
        echo "❌ 無效選項！"
        exit 1
        ;;
esac

