# Isaac Lab v2.2 API 變更指南

## 🚨 重要變更：模組重命名

在 Isaac Lab v2.2 中，為了支援 Isaac Sim 4.5/5.0，所有模組名稱進行了重大重命名。

### 📋 官方說明

根據 Isaac Lab v2.2 Release Notes：

> "Renaming of Isaac Lab Extensions and Folders:
> `omni.isaac.lab` → `isaaclab`"

**參考資料**:
- [Isaac Lab v2.2.0 Release Notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.2.0)
- [Isaac Lab Documentation - Migration Guide](https://isaac-sim.github.io/IsaacLab/)

## 🔄 模組名稱對照表

| 舊名稱 (Isaac Lab < v2.2) | 新名稱 (Isaac Lab v2.2+) |
|---------------------------|-------------------------|
| `omni.isaac.lab` | `isaaclab` |
| `omni.isaac.lab.utils` | `isaaclab.utils` |
| `omni.isaac.lab.sim` | `isaaclab.sim` |
| `omni.isaac.lab.envs` | `isaaclab.envs` |
| `omni.isaac.lab.managers` | `isaaclab.managers` |
| `omni.isaac.lab.assets` | `isaaclab.assets` |
| `omni.isaac.lab.sensors` | `isaaclab.sensors` |
| `omni.isaac.lab.controllers` | `isaaclab.controllers` |

## ✅ 修正示例

### 錯誤的導入 (舊版)

```python
# ❌ 錯誤 - Isaac Lab v2.2 不再支援
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import RewardTermCfg, SceneEntityCfg
```

### 正確的導入 (新版)

```python
# ✅ 正確 - Isaac Lab v2.2+
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
```

## 🔍 如何檢查和修正

### 1. 搜索所有使用舊模組的地方

```bash
# 在項目中搜索所有 omni.isaac.lab 的使用
grep -r "omni\.isaac\.lab" source/ --include="*.py"

# 或使用更詳細的搜索
find source/ -name "*.py" -exec grep -H "omni\.isaac\.lab" {} \;
```

### 2. 批量替換（謹慎使用）

```bash
# 在特定目錄下批量替換
find source/isaaclab_tasks/ -name "*.py" -exec sed -i 's/omni\.isaac\.lab/isaaclab/g' {} \;
```

⚠️ **注意**: 批量替換前請先備份，並檢查替換結果！

### 3. 手動替換（推薦）

對於每個文件，手動檢查並替換，確保正確性：

```python
# 1. 打開文件
# 2. 查找所有 "omni.isaac.lab" 
# 3. 替換為 "isaaclab"
# 4. 確認 import 仍然正確
# 5. 測試運行
```

## 🧪 驗證修正

### 測試導入

```python
# 測試新模組是否可以正確導入
./isaaclab.sh -p -c "from isaaclab.utils import configclass; print('✅ isaaclab.utils 導入成功')"

./isaaclab.sh -p -c "from isaaclab.sim import SimulationCfg; print('✅ isaaclab.sim 導入成功')"

./isaaclab.sh -p -c "from isaaclab.envs import ManagerBasedRLEnv; print('✅ isaaclab.envs 導入成功')"
```

### 運行環境註冊測試

```bash
./isaaclab.sh -p register_local_planner.py
```

應該看到：
```
✅ Nova Carter 本地規劃器環境已手動註冊
✅ 環境註冊驗證成功
```

### 運行簡單訓練測試

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-Easy-v0 \
    --num_envs 2 \
    --headless \
    --max_iterations 10
```

## 📊 我們的修正記錄

### 已修正的文件

| 文件 | 修正內容 | 狀態 |
|------|----------|------|
| `local_planner_env_cfg_gui_fixed.py` | `omni.isaac.lab.utils` → `isaaclab.utils`<br>`omni.isaac.lab.sim` → `isaaclab.sim` | ✅ 完成 |
| `local_planner_env_cfg_easy.py` | `omni.isaac.lab.utils` → `isaaclab.utils`<br>`omni.isaac.lab.managers` → `isaaclab.managers` | ✅ 完成 |

### 其他 Isaac Lab 項目文件

其他非我們創建的文件（如 `cartpole`, `anymal` 等）可能仍使用舊名稱，但這些不影響我們的 local planner 環境。

## 🔧 常見錯誤和解決方案

### 錯誤 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'omni.isaac.lab'
```

**原因**: 使用了舊的模組名稱

**解決**: 將所有 `omni.isaac.lab` 替換為 `isaaclab`

### 錯誤 2: ImportError

```
ImportError: cannot import name 'configclass' from 'omni.isaac.lab.utils'
```

**原因**: 模組路徑錯誤

**解決**: 
```python
# 從這樣
from omni.isaac.lab.utils import configclass

# 改為這樣
from isaaclab.utils import configclass
```

### 錯誤 3: 混合使用新舊模組

```python
# ❌ 錯誤 - 混合使用
from isaaclab.utils import configclass
from omni.isaac.lab.sim import SimulationCfg  # 舊的

# ✅ 正確 - 統一使用新名稱
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
```

## 💡 最佳實踐

1. **統一使用新模組名稱**: 在所有新代碼中使用 `isaaclab.*`
2. **檢查依賴**: 確保所有依賴的模組也使用新名稱
3. **測試驗證**: 修改後立即測試導入和基本功能
4. **文檔更新**: 更新所有相關文檔中的模組名稱
5. **代碼審查**: 在提交前檢查是否還有遺漏的舊模組名稱

## 📚 參考資源

- [Isaac Lab v2.2 Release Notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.2.0)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Migration Guide](https://isaac-sim.github.io/IsaacLab/main/migration.html)

## ✅ 檢查清單

在修正完成後，確認以下項目：

- [ ] 搜索並修正所有 `omni.isaac.lab` 導入
- [ ] 測試所有修正的文件可以正確導入
- [ ] 運行環境註冊測試
- [ ] 運行簡單的訓練測試
- [ ] 更新相關文檔
- [ ] 提交修正到版本控制

---

**記住**: Isaac Lab v2.2+ 完全不再支援 `omni.isaac.lab` 前綴，必須使用 `isaaclab`！
