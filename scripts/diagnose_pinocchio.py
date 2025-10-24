#!/usr/bin/env python3
"""診斷 Pinocchio 依賴問題"""

import sys

print("=" * 80)
print("Pinocchio 依賴診斷")
print("=" * 80 + "\n")

# 測試 1: 檢查 Pinocchio 是否安裝
print("1. 檢查 Pinocchio 是否安裝...")
try:
    import pinocchio
    print(f"   ✓ Pinocchio 已安裝（版本: {pinocchio.__version__}）")
    print(f"   警告: 這可能導致衝突")
except ImportError:
    print("   ✓ Pinocchio 未安裝（這是好事！）")

# 測試 2: 檢查 hpp-fcl
print("\n2. 檢查 hpp-fcl...")
try:
    import hppfcl
    print(f"   ✓ hpp-fcl 已安裝")
    print(f"   警告: 這可能導致衝突")
except ImportError:
    print("   ✓ hpp-fcl 未安裝（這是好事！）")

# 測試 3: 嘗試導入我們的環境
print("\n3. 測試導入我們的環境...")
try:
    from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg import LocalPlannerEnvCfg
    print("   ✓ 成功導入 LocalPlannerEnvCfg")
except Exception as e:
    print(f"   ✗ 導入失敗: {e}")
    sys.exit(1)

# 測試 4: 檢查環境是否使用 Pinocchio
print("\n4. 檢查環境配置...")
cfg = LocalPlannerEnvCfg()
has_pinocchio = False

# 檢查動作配置
if hasattr(cfg, 'actions'):
    for action_name in dir(cfg.actions):
        if 'pink' in action_name.lower() or 'pinocchio' in action_name.lower():
            has_pinocchio = True
            print(f"   警告: 動作配置中發現 Pinocchio 相關: {action_name}")

if not has_pinocchio:
    print("   ✓ 環境配置中無 Pinocchio 依賴")

# 測試 5: 列出黑名單
print("\n5. 檢查 isaaclab_tasks 黑名單...")
try:
    from isaaclab_tasks import _BLACKLIST_PKGS
    print(f"   當前黑名單: {_BLACKLIST_PKGS}")
    if "pinocchio" in str(_BLACKLIST_PKGS) or "pick_place" in _BLACKLIST_PKGS:
        print("   ✓ Pinocchio 相關模組已被排除")
    else:
        print("   警告: 黑名單中未包含 Pinocchio 相關模組")
except Exception as e:
    print(f"   無法讀取黑名單: {e}")

print("\n" + "=" * 80)
print("診斷完成")
print("=" * 80)
print("\n建議：")
print("- 如果看到 '✓'，表示該項檢查通過")
print("- 如果看到 '警告' 或 '✗'，可能需要進一步處理")
print("\n如果所有檢查都通過，請嘗試運行:")
print("  ./isaaclab.sh -p scripts/test_local_planner_env.py")





