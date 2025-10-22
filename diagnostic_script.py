#!/usr/bin/env python3
"""
系統性診斷腳本：Isaac Sim 模組問題根因分析
按照用戶建議進行逐步排查
"""

print('=== 🔍 系統性診斷：Isaac Sim 模組問題根因分析 ===')

import sys
import importlib
import os
import glob

print(f'1️⃣ Python 執行環境檢查：')
print(f'   Python 執行器: {sys.executable}')
print(f'   Python 版本: {sys.version}')
print(f'   是否為 Isaac Sim Python: {"/isaac" in sys.executable or "/isaacsim" in sys.executable}')

print(f'\n2️⃣ 模組路徑檢查：')
isaac_sim_paths = [p for p in sys.path if 'isaac' in p.lower()]
print(f'   Isaac 相關路徑數量: {len(isaac_sim_paths)}')
for i, path in enumerate(isaac_sim_paths[:5]):  # 只顯示前5個
    print(f'   [{i+1}] {path}')

print(f'\n3️⃣ omni.isaac.core 模組檢查：')
try:
    m = importlib.import_module('omni.isaac.core')
    print(f'   ✅ omni.isaac.core 找到於: {m.__file__}')
    
    # 檢查子模組
    try:
        m_utils = importlib.import_module('omni.isaac.core.utils')
        print(f'   ✅ omni.isaac.core.utils 可用')
        
        try:
            m_torch = importlib.import_module('omni.isaac.core.utils.torch')
            print(f'   ✅ omni.isaac.core.utils.torch 找到於: {m_torch.__file__}')
            
            # 檢查 set_cuda_device 函數
            if hasattr(m_torch, 'set_cuda_device'):
                print(f'   ✅ set_cuda_device 函數可用')
            else:
                print(f'   ❌ set_cuda_device 函數不存在')
                print(f'   📋 可用函數: {[attr for attr in dir(m_torch) if not attr.startswith("_")]}')
                
        except ImportError as e:
            print(f'   ❌ omni.isaac.core.utils.torch 導入失敗: {e}')
    except ImportError as e:
        print(f'   ❌ omni.isaac.core.utils 導入失敗: {e}')
        
except ImportError as e:
    print(f'   ❌ omni.isaac.core 導入失敗: {e}')

print(f'\n4️⃣ 新版模組檢查 (isaacsim.*)：')
new_modules = [
    'isaacsim.core',
    'isaacsim.core.api',
    'isaacsim.core.api.utils',
    'isaacsim.core.api.utils.torch',
    'isaacsim.core.utils.torch'
]

for module_name in new_modules:
    try:
        m = importlib.import_module(module_name)
        print(f'   ✅ {module_name} 找到於: {getattr(m, "__file__", "內建模組")}')
        if 'torch' in module_name and hasattr(m, 'set_cuda_device'):
            print(f'      → set_cuda_device 函數可用')
    except ImportError:
        print(f'   ❌ {module_name} 不存在')

print('\n=== 🔍 擴展和安裝完整性檢查 ===')

print('5️⃣ Isaac Sim 安裝結構檢查：')
isaac_sim_path = '/home/aa/isaacsim'
if os.path.exists(isaac_sim_path):
    print(f'   Isaac Sim 路徑: {isaac_sim_path}')
    
    # 檢查關鍵目錄
    key_dirs = ['exts', 'extsDeprecated', 'kit', 'python_packages']
    for dir_name in key_dirs:
        dir_path = os.path.join(isaac_sim_path, dir_name)
        exists = os.path.exists(dir_path)
        print(f'   {dir_name}/: {"✅存在" if exists else "❌缺失"}')
        
        if exists and dir_name == 'exts':
            # 檢查 exts 目錄中的 isaac 相關擴展
            isaac_exts = glob.glob(os.path.join(dir_path, '*isaac*'))
            print(f'      Isaac擴展數量: {len(isaac_exts)}')
            for ext in isaac_exts[:3]:  # 只顯示前3個
                print(f'      - {os.path.basename(ext)}')
                
        if exists and dir_name == 'extsDeprecated':
            # 檢查 extsDeprecated 目錄
            deprecated_exts = glob.glob(os.path.join(dir_path, '*isaac*'))
            print(f'      已棄用Isaac擴展數量: {len(deprecated_exts)}')
            for ext in deprecated_exts[:3]:
                print(f'      - {os.path.basename(ext)}')
else:
    print(f'   ❌ Isaac Sim 路徑不存在: {isaac_sim_path}')

print(f'\n6️⃣ Kit Python 環境檢查：')
kit_python_path = '/home/aa/isaacsim/kit/python'
if os.path.exists(kit_python_path):
    print(f'   Kit Python 路徑: ✅存在')
    
    # 檢查 site-packages
    site_packages = os.path.join(kit_python_path, 'lib/python3.11/site-packages')
    if os.path.exists(site_packages):
        print(f'   site-packages: ✅存在')
        
        # 尋找 omni 相關包
        omni_packages = glob.glob(os.path.join(site_packages, '*omni*'))
        print(f'   omni 相關包數量: {len(omni_packages)}')
        
        # 特別檢查 omni.isaac
        isaac_packages = [p for p in omni_packages if 'isaac' in p]
        print(f'   omni.isaac 相關包數量: {len(isaac_packages)}')
        for pkg in isaac_packages[:5]:
            print(f'      - {os.path.basename(pkg)}')
    else:
        print(f'   ❌ site-packages 不存在')
else:
    print(f'   ❌ Kit Python 路徑不存在')

print(f'\n7️⃣ 環境變數檢查：')
env_vars = ['PYTHONPATH', 'ISAAC_SIM_ROOT', 'OMNI_KIT_ROOT']
for var in env_vars:
    value = os.environ.get(var, 'Not Set')
    print(f'   {var}: {value}')

print('\n=== 🔍 深度模組載入測試 ===')

print('8️⃣ 手動路徑添加測試：')

# 添加可能的模組路徑
potential_paths = [
    '/home/aa/isaacsim/exts',
    '/home/aa/isaacsim/extsDeprecated',  
    '/home/aa/isaacsim/kit/python/lib/python3.11/site-packages',
    '/home/aa/isaacsim/python_packages'
]

for path in potential_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f'   已添加路徑: {path}')

print(f'\n9️⃣ 重新測試模組導入：')
try:
    import omni.isaac.core
    print(f'   ✅ 添加路徑後，omni.isaac.core 可導入')
    print(f'   模組路徑: {omni.isaac.core.__file__}')
    
    try:
        from omni.isaac.core.utils.torch import set_cuda_device
        print(f'   ✅ set_cuda_device 成功導入')
        
        # 嘗試調用
        try:
            set_cuda_device(0)
            print(f'   ✅ set_cuda_device(0) 調用成功')
        except Exception as e:
            print(f'   ⚠️ set_cuda_device(0) 調用失敗: {e}')
            
    except ImportError as e:
        print(f'   ❌ set_cuda_device 導入失敗: {e}')
        
except ImportError as e:
    print(f'   ❌ 即使添加路徑，omni.isaac.core 仍無法導入: {e}')

print(f'\n🔟 依賴檢查：')
dependencies = ['carb', 'omni.kit', 'pxr']
for dep in dependencies:
    try:
        importlib.import_module(dep)
        print(f'   ✅ {dep} 模組可用')
    except ImportError:
        print(f'   ❌ {dep} 模組缺失（這可能是根本原因）')

print('\n=== 📋 診斷總結 ===')
print('根據以上檢查，請查看哪些模組缺失或路徑不正確')
print('這將幫助我們確定問題的真正根源')
