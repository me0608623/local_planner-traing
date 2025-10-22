#!/usr/bin/env python3
"""
PhysX Tensor Device 診斷工具

基於您提到的NVIDIA開發者論壇和GitHub issue分析，
此腳本專門診斷 Isaac Sim/Lab 中的張量設備不一致問題。

參考資料：
- NVIDIA Developer Forums: PhysX tensors device mismatch
- Isaac Lab GitHub Issues: Incompatible device of velocity tensor
"""

import sys
import torch
import argparse

def diagnose_simulation_setup():
    """診斷模擬設置的設備一致性"""
    print("=== 🔍 PhysX Tensor Device 診斷工具 ===")
    print("基於 NVIDIA 開發者論壇和 GitHub 已知問題分析\n")
    
    print("1️⃣ 系統環境檢查：")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 設備數: {torch.cuda.device_count()}")
        print(f"   當前 CUDA 設備: {torch.cuda.current_device()}")
    
    print(f"\n2️⃣ Isaac Sim/Lab 模組檢查：")
    try:
        import omni
        print("   ✅ omni 模組可用")
        
        try:
            import omni.physx.tensors
            print("   ✅ omni.physx.tensors 可用")
        except ImportError as e:
            print(f"   ❌ omni.physx.tensors 不可用: {e}")
            
        try:
            from omni.isaac.core.utils.torch import set_cuda_device
            print("   ✅ omni.isaac.core.utils.torch 可用")
        except ImportError:
            try:
                from isaacsim.core.api.utils.torch import set_cuda_device
                print("   ✅ isaacsim.core.api.utils.torch 可用 (Isaac Sim 5.0)")
            except ImportError as e:
                print(f"   ❌ Isaac Sim torch utils 不可用: {e}")
                
    except ImportError as e:
        print(f"   ❌ Isaac Sim 模組不可用: {e}")
    
    print(f"\n3️⃣ 張量設備測試：")
    if torch.cuda.is_available():
        # 測試不同設備上的張量
        cpu_tensor = torch.randn(100, 3)
        gpu_tensor = torch.randn(100, 3).cuda()
        
        print(f"   CPU 張量設備: {cpu_tensor.device}")
        print(f"   GPU 張量設備: {gpu_tensor.device}")
        
        # 模擬PhysX可能遇到的設備不匹配
        try:
            mixed_result = cpu_tensor + gpu_tensor.cpu()
            print("   ✅ 混合張量操作成功")
        except RuntimeError as e:
            print(f"   ⚠️ 混合張量操作警告: {e}")
    
    print(f"\n4️⃣ 建議的修復方案：")
    print("   - 確保 SimulationCfg.device = 'cuda:0' (如果使用GPU)")
    print("   - 增加 PhysX GPU 緩衝區大小")
    print("   - 使用我們的 LocalPlannerEnvCfg_GPU_FIXED 配置")
    print("   - 如果問題持續，嘗試 CPU 模式作為workaround")

def test_environment_registration():
    """測試環境註冊和設備配置"""
    print(f"\n5️⃣ 環境註冊測試：")
    
    try:
        import gymnasium as gym
        
        # 測試我們的修復版本環境
        env_configs = [
            "Isaac-Navigation-LocalPlanner-Carter-v0",
            "Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0", 
            "Isaac-Navigation-LocalPlanner-Carter-CPU-v0",
            "Isaac-Navigation-LocalPlanner-Carter-IsaacSim5-v0"
        ]
        
        for env_name in env_configs:
            try:
                spec = gym.spec(env_name)
                if spec:
                    print(f"   ✅ {env_name} 已註冊")
                else:
                    print(f"   ❌ {env_name} 未註冊")
            except Exception as e:
                print(f"   ⚠️ {env_name} 註冊檢查失敗: {e}")
                
    except ImportError as e:
        print(f"   ❌ gymnasium 不可用: {e}")

def main():
    parser = argparse.ArgumentParser(description="PhysX Tensor Device 診斷工具")
    parser.add_argument("--full", action="store_true", help="運行完整診斷")
    args = parser.parse_args()
    
    diagnose_simulation_setup()
    
    if args.full:
        test_environment_registration()
    
    print(f"\n📚 相關資源：")
    print("   - NVIDIA Developer Forums: PhysX tensors device issues")
    print("   - Isaac Lab GitHub Issues: tensor device mismatch")
    print("   - 我們的修復文檔: md/PHYSX_TENSOR_DEVICE_FIX.md")
    
    print(f"\n🎯 推薦使用：")
    print("   python scripts/reinforcement_learning/rsl_rl/train.py \\")
    print("       --task Isaac-Navigation-LocalPlanner-Carter-GPU-Fixed-v0 \\")
    print("       --num_envs 4 --headless")

if __name__ == "__main__":
    main()
