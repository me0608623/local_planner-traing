#!/usr/bin/env python3
"""
測試原始環境是否還能正常工作
"""

print("=" * 80)
print("🧪 測試原始環境是否受影響")
print("=" * 80)

# 測試1: 導入基本模組
print("\n1️⃣ 測試基本模組導入...")
try:
    from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg import LocalPlannerEnvCfg
    print("✅ 原始 LocalPlannerEnvCfg 導入成功")
except Exception as e:
    print(f"❌ 原始 LocalPlannerEnvCfg 導入失敗: {e}")

# 測試2: 測試新文件導入（可能有問題）
print("\n2️⃣ 測試新添加的文件導入...")
try:
    from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg_gui_fixed import LocalPlannerEnvCfg_GUI_FIXED
    print("✅ LocalPlannerEnvCfg_GUI_FIXED 導入成功")
except Exception as e:
    print(f"❌ LocalPlannerEnvCfg_GUI_FIXED 導入失敗: {e}")

try:
    from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg_easy import LocalPlannerEnvCfg_EASY
    print("✅ LocalPlannerEnvCfg_EASY 導入成功")
except Exception as e:
    print(f"❌ LocalPlannerEnvCfg_EASY 導入失敗: {e}")

# 測試3: 環境註冊
print("\n3️⃣ 測試環境註冊...")
try:
    import gymnasium as gym
    spec = gym.spec('Isaac-Navigation-LocalPlanner-Carter-v0')
    print(f"✅ 原始環境 Isaac-Navigation-LocalPlanner-Carter-v0 註冊成功")
    print(f"   Entry point: {spec.entry_point}")
except Exception as e:
    print(f"❌ 環境註冊失敗: {e}")

print("\n" + "=" * 80)
print("如果看到 ❌ 說明新添加的文件導入失敗，影響了整個模組")
print("=" * 80)
