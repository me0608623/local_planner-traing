#!/bin/bash
# 設置環境變數以修復 Pinocchio/Assimp 衝突
# 
# 使用方法：
#   source setup_env_fix_pinocchio.sh
#   然後再運行 isaaclab.sh

# 將 Isaac Sim 的庫路徑放在最前面
export LD_LIBRARY_PATH="$HOME/IsaacLab/_isaac_sim/kit/lib:$LD_LIBRARY_PATH"

echo "✓ 已設置環境變數修復 Pinocchio/Assimp 衝突"
echo "  LD_LIBRARY_PATH 已更新"
echo ""
echo "現在可以運行："
echo "  ./isaaclab.sh -p scripts/test_local_planner_env.py"




