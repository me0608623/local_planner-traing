#!/usr/bin/env python3
"""
測試 Isaac Lab v2.2 新模組導入

⚠️ 使用方法:
    ./isaaclab.sh -p test_module_imports.py
"""

print("=" * 80)
print("🧪 測試 Isaac Lab v2.2 模組導入")
print("=" * 80)

# 測試核心模組
try:
    from isaaclab.utils import configclass
    print("✅ isaaclab.utils.configclass 導入成功")
except ImportError as e:
    print(f"❌ isaaclab.utils 導入失敗: {e}")

try:
    from isaaclab.sim import SimulationCfg
    print("✅ isaaclab.sim.SimulationCfg 導入成功")
except ImportError as e:
    print(f"❌ isaaclab.sim 導入失敗: {e}")

try:
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    print("✅ isaaclab.managers.RewardTermCfg 導入成功")
    print("✅ isaaclab.managers.SceneEntityCfg 導入成功")
except ImportError as e:
    print(f"❌ isaaclab.managers 導入失敗: {e}")

try:
    from isaaclab.envs import ManagerBasedRLEnv
    print("✅ isaaclab.envs.ManagerBasedRLEnv 導入成功")
except ImportError as e:
    print(f"❌ isaaclab.envs 導入失敗: {e}")

# 測試我們修正的配置文件
print("\n" + "=" * 80)
print("🧪 測試修正後的配置文件導入")
print("=" * 80)

try:
    from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg_gui_fixed import (
        LocalPlannerEnvCfg_GUI_FIXED
    )
    print("✅ LocalPlannerEnvCfg_GUI_FIXED 導入成功")
except ImportError as e:
    print(f"❌ LocalPlannerEnvCfg_GUI_FIXED 導入失敗: {e}")

try:
    from isaaclab_tasks.manager_based.navigation.local_planner.local_planner_env_cfg_easy import (
        LocalPlannerEnvCfg_EASY
    )
    print("✅ LocalPlannerEnvCfg_EASY 導入成功")
except ImportError as e:
    print(f"❌ LocalPlannerEnvCfg_EASY 導入失敗: {e}")

# 測試環境註冊
print("\n" + "=" * 80)
print("🧪 測試環境註冊")
print("=" * 80)

try:
    import gymnasium as gym
    
    # 測試我們的環境是否可以被找到
    envs_to_test = [
        "Isaac-Navigation-LocalPlanner-Carter-Easy-v0",
        "Isaac-Navigation-LocalPlanner-Carter-GUI-Fixed-v0",
        "Isaac-Navigation-LocalPlanner-Carter-Curriculum-Stage1-v0",
    ]
    
    for env_id in envs_to_test:
        try:
            spec = gym.spec(env_id)
            print(f"✅ {env_id} 註冊成功")
        except Exception as e:
            print(f"⚠️ {env_id} 尚未註冊或找不到")
            
except Exception as e:
    print(f"❌ 環境註冊測試失敗: {e}")

print("\n" + "=" * 80)
print("🎉 測試完成！")
print("=" * 80)
print("\n如果所有核心模組都顯示 ✅，表示 API 變更修正成功！")
print("如果環境顯示 ⚠️，請運行：")
print("  ./isaaclab.sh -p register_local_planner.py")
print("=" * 80)
