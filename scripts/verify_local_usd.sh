#!/bin/bash
# 驗證本地 USD 檔案設定

echo "=========================================="
echo "驗證本地 USD 檔案設定"
echo "=========================================="
echo ""

# 1. 檢查 USD 檔案
echo "1. 檢查 USD 檔案..."
USD_PATH="/home/aa/isaacsim/usd/nova_carter.usd"

if [ -f "$USD_PATH" ]; then
    echo "   ✓ USD 檔案存在: $USD_PATH"
    FILE_SIZE=$(ls -lh "$USD_PATH" | awk '{print $5}')
    echo "   ✓ 檔案大小: $FILE_SIZE"
else
    echo "   ✗ 錯誤: USD 檔案不存在!"
    echo "   預期位置: $USD_PATH"
    exit 1
fi

# 2. 檢查配置文件
echo ""
echo "2. 檢查配置文件..."
CONFIG_FILE="source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/local_planner_env_cfg.py"

if grep -q "/home/aa/isaacsim/usd/nova_carter.usd" "$CONFIG_FILE"; then
    echo "   ✓ 配置文件已更新為本地路徑"
else
    echo "   ✗ 警告: 配置文件可能未正確更新"
    echo "   請檢查: $CONFIG_FILE"
fi

# 3. 檢查配置完整性
echo ""
echo "3. 檢查配置完整性..."

if grep -q "actuators=" "$CONFIG_FILE"; then
    echo "   ✓ actuators 配置存在"
else
    echo "   ✗ 警告: 缺少 actuators 配置"
fi

if grep -q 'body_name="chassis_link"' "$CONFIG_FILE"; then
    echo "   ✓ body_name 配置存在"
else
    echo "   ✗ 警告: 缺少 body_name 配置"
fi

# 4. 檢查 Pinocchio
echo ""
echo "4. 檢查 Pinocchio 狀態..."
if $HOME/IsaacLab/_isaac_sim/python.sh -c "import pinocchio" 2>/dev/null; then
    echo "   ⚠ Pinocchio 仍然安裝"
    echo "   建議運行: ./scripts/remove_pinocchio.sh"
else
    echo "   ✓ Pinocchio 已移除或未安裝"
fi

# 5. 檢查 controllers 修復
echo ""
echo "5. 檢查 controllers 修復..."
if grep -q "try:" source/isaaclab/isaaclab/controllers/__init__.py | head -1; then
    echo "   ✓ controllers/__init__.py 已修改（pink_ik 可選）"
else
    echo "   ⚠ controllers/__init__.py 可能未修改"
fi

# 總結
echo ""
echo "=========================================="
echo "驗證完成"
echo "=========================================="
echo ""
echo "建議下一步："
echo "  1. 如果所有檢查都通過，運行測試："
echo "     ./isaaclab.sh -p scripts/test_local_planner_env.py"
echo ""
echo "  2. 查看完整日誌確認從本地載入："
echo "     應該看到: [INFO] Spawning asset from: $USD_PATH"
echo ""



