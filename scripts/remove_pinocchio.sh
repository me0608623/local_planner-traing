#!/bin/bash
# 移除 Pinocchio 及相關套件（我們不需要它們）

echo "=================================================="
echo "移除 Pinocchio 及相關套件"
echo "=================================================="
echo ""

# 使用 Isaac Sim 的 Python
PYTHON="$HOME/IsaacLab/_isaac_sim/python.sh"

echo "檢查 Pinocchio 是否已安裝..."
$PYTHON -c "import pinocchio; print(f'已安裝: {pinocchio.__version__}')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "移除 Pinocchio..."
    $PYTHON -m pip uninstall -y pinocchio pin
    
    echo ""
    echo "移除 hpp-fcl..."
    $PYTHON -m pip uninstall -y hpp-fcl
    
    echo ""
    echo "移除 cmeel..."
    $PYTHON -m pip uninstall -y cmeel cmeel-assimp
    
    echo ""
    echo "✓ 移除完成！"
else
    echo "✓ Pinocchio 未安裝"
fi

echo ""
echo "驗證..."
$PYTHON -c "import pinocchio" 2>/dev/null && echo "警告: Pinocchio 仍然存在" || echo "✓ Pinocchio 已成功移除"

echo ""
echo "=================================================="
echo "完成！現在可以運行："
echo "  ./isaaclab.sh -p scripts/test_local_planner_env.py"
echo "=================================================="




