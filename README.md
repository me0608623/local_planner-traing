# Nova Carter Local Planner - 強化學習環境

基於 Isaac Lab 和 Isaac Sim 5.0 的 Nova Carter 機器人本地路徑規劃強化學習環境。

## ⚠️ 重要提醒

### 1. 使用正確的 Python 環境

**所有命令必須使用 `./isaaclab.sh -p` 而不是系統 `python`！**

Isaac Lab 需要特定的 Python 環境和依賴，直接使用系統 Python 會導致模組導入錯誤。

```bash
# ✅ 正確
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...

# ❌ 錯誤
python scripts/reinforcement_learning/rsl_rl/train.py ...
```

### 2. Isaac Lab v2.2 API 變更

**Isaac Lab v2.2 已將所有模組從 `omni.isaac.lab` 重命名為 `isaaclab`！**

```python
# ✅ 正確 - Isaac Lab v2.2+
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

# ❌ 錯誤 - 舊版本（不再支援）
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim import SimulationCfg
```

---

## 📚 文檔導航

### 🎯 快速開始

**新手入門**（依序閱讀）：
1. **本 README** - 專案概述和環境設置
2. **[指令快速參考](md/指令快速參考.md)** - 常用訓練指令
3. **[訓練架構完整說明](md/訓練架構完整說明.md)** - 深入理解系統設計

### 📖 完整文檔列表

1. **[訓練架構完整說明](md/訓練架構完整說明.md)** 🏗️（主要技術文檔）
   - PCCBF-MPC 論文整合
   - 觀測、動作、獎勵設計
   - 課程學習策略
   - 訓練參數配置

2. **[指令快速參考](md/指令快速參考.md)** 🚀（命令速查表）
   - 訓練指令範例
   - 測試和評估指令
   - 常見問題排除

3. **[訓練流程詳解](md/訓練流程詳解.md)** 🔄（技術深入）
   - train.py 逐行註解
   - PPO 算法原理
   - 訓練數學公式

4. **[實驗記錄與改進總結](md/最新改進總結.md)** 📊（實驗追蹤）
   - 歷次訓練實驗結果
   - 成功/失敗案例分析
   - 關鍵發現和教訓

5. **[PCCBF 訓練測試指南](md/PCCBF_訓練測試指南.md)** 🧪（進階參考）
   - 測試流程和驗證方法
   - 故障排除指南
   - 效能評估標準

---

## 🎉 訓練成果亮點

**最新成果**（2025-10-27）：
- ✅ 成功修正獎勵函數，訓練從完全失敗（-10062獎勵）提升到優秀（+27.67獎勵）
- ✅ **DEBUG配置達成 37.50% 成功率**（0.3-1.0米導航，5000 iterations）⭐
- ✅ **位置精度 0.72米，0% 碰撞率**（安全且精確）
- ✅ 發現極簡獎勵設計（3項）優於複雜設計（7項）：**成功率差距 18倍**（37.5% vs 2%）
- ✅ 證明深化訓練有效（5000 iterations 比 1000 iterations 成功率翻倍）

**核心方法**：
- 極簡獎勵（progress_to_goal + reached_goal + standstill_penalty）
- 高權重設計（50, 500）
- 溫和的課程學習（每次難度增加 < 50%）

詳見：[實驗記錄與改進總結](md/最新改進總結.md)

---

## 🚀 快速開始訓練

### 推薦配置：DEBUG（已驗證成功）

**特性**：
- 成功率：18.75%
- 目標距離：0.3-1.0米
- 極簡獎勵設計

**訓練指令**：
```bash
cd /home/aa/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
    --num_envs 16 \
    --max_iterations 1000 \
    --headless
```

更多配置請參考：[指令快速參考](md/指令快速參考.md)

---

## 🚀 專案簡介

本專案實現了 Nova Carter 移動機器人的本地路徑規劃強化學習環境，支援障礙物迴避、目標導航和 LiDAR 感知。環境完全兼容 Isaac Sim 5.0，並修復了所有已知的 PhysX tensor device 匹配問題。

### ✨ 主要特性

- 🤖 **Nova Carter 機器人模擬**：完整的差動驅動機器人模型
- 🎯 **目標導航任務**：智能體需學習從起點導航到目標點
- 👁️ **LiDAR 感知**：360度雷射雷達感測器進行環境感知（每度1條射線，共360條）
- 🚧 **障礙物環境**：隨機生成的靜態障礙物
- ⚡ **多環境並行**：支援 48 個環境同時訓練
- 🔧 **Isaac Sim 5.0 兼容**：完全支援最新版本的 Isaac Sim

---

## 📋 環境要求

- **Isaac Sim 5.0+**
- **Isaac Lab 2.2+**
- **Python 3.11**
- **CUDA 12.x** (GPU 模式)
- **RSL-RL** 強化學習庫

---

## 🛠️ 安裝和設置

### 1. 環境準備

```bash
# 確保在正確的 conda 環境中
conda activate env_isaaclab

# 安裝必要的 Python 依賴
pip install packaging

# 確認 Isaac Sim 路徑
ls -la _isaac_sim  # 應該指向您的 Isaac Sim 安裝目錄
```

### 2. 環境註冊

環境會在導入時自動註冊，或可手動註冊：

```bash
./isaaclab.sh -p register_local_planner.py
```

### 3. 驗證環境

```bash
# 列出可用環境
./isaaclab.sh -p scripts/tools/list_envs.py | grep Carter

# 應該看到：Isaac-Navigation-LocalPlanner-Carter-v0
```

---

## 🎮 快速開始訓練

### 訓練指令（推薦使用DEBUG配置）⭐

```bash
cd /home/aa/IsaacLab

# DEBUG配置（已驗證：18.75%成功率）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-DEBUG-v0 \
    --num_envs 16 \
    --max_iterations 1000 \
    --headless
```

**訓練時間**：約 20 分鐘（RTX 3090/4090/5090，1000 次迭代）

### 測試訓練好的模型

```bash
# 測試模型（GUI 可視化）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/local_planner_carter/[日期]/model_2999.pt
```

### 觀察訓練過程

```bash
# 啟動 TensorBoard 查看訓練曲線
tensorboard --logdir logs/rsl_rl/

# 瀏覽器打開：http://localhost:6006
```

**📖 更多詳細指令請參閱：[指令快速參考](md/指令快速參考.md)**

---

## 📊 訓練架構概覽

### 觀測空間 (369 維)

- **LiDAR 距離** [360]：360° 雷射掃描數據
- **機器人線速度** [3]：`[vx, vy, vz]` 在機器人座標系
- **機器人角速度** [3]：`[wx, wy, wz]` 在機器人座標系
- **目標相對位置** [2]：`[dx, dy]` 在機器人座標系
- **到目標距離** [1]：歐幾里得距離

### 動作空間 (2 維)

- **線速度指令** [1]：-2.0 到 +2.0 m/s
- **角速度指令** [1]：-π 到 +π rad/s

### 獎勵函數（極簡設計 - 已驗證有效）

```
總獎勵 = 
  + progress_to_goal × 50.0        # 接近目標（主要驅動力）
  + reached_goal × 500.0           # 到達目標（成功大獎）
  - standstill × 0.1               # 靜止不動（防止卡住）
```

**設計理念**：
- ✅ 只用3個獎勵項（極簡設計）
- ✅ 高權重正向獎勵（50, 500）
- ❌ 移除複雜的懲罰項（已驗證：簡單更有效）

**成果**：極簡設計達成 **18.75%** 成功率，複雜設計僅 < 2%

### Agent 架構

**演算法**：PPO (Proximal Policy Optimization)

**神經網路**：
- Actor Network: `[369] → [256] → [256] → [128] → [2]`
- Critic Network: `[369] → [256] → [256] → [128] → [1]`
- 激活函數：ELU
- 總參數量：約 250K-300K

**📖 詳細架構說明請參閱：[訓練架構完整說明](md/訓練架構完整說明.md)**

---

## 📁 代碼結構

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/local_planner/
├─ local_planner_env_cfg.py          # 環境配置（主文件）⭐
├─ __init__.py                       # 環境註冊
├─ agents/
│  ├─ rsl_rl_ppo_cfg.py             # PPO 配置 ⭐
│  └─ sb3_ppo_cfg.py                # Stable Baselines3 配置
└─ mdp/                              # MDP 組件實現 ⭐
   ├─ observations.py               # 觀測函數實現
   ├─ actions.py                    # 動作轉換
   ├─ rewards.py                    # 獎勵計算
   └─ terminations.py               # 終止條件

scripts/reinforcement_learning/rsl_rl/
├─ train.py                          # 訓練腳本 ⭐
└─ play.py                           # 測試腳本
```

---

## 🎯 訓練結果位置

訓練完成後，結果保存在：

```
logs/rsl_rl/local_planner_carter/
└─ [訓練日期時間]/
   ├─ model_0.pt                # 初始模型
   ├─ model_100.pt              # 第 100 次迭代
   ├─ ...
   ├─ model_2999.pt             # 最終模型
   ├─ events.out.tfevents.*     # TensorBoard 日誌
   └─ params/                   # 訓練配置
      ├─ env.json
      └─ agent.json
```

---

## 🔧 常見問題

### 1. 環境無法註冊

```bash
# 手動註冊環境
./isaaclab.sh -p register_local_planner.py

# 驗證
./isaaclab.sh -p scripts/tools/list_envs.py | grep Carter
```

### 2. 模組導入錯誤

確保使用 `./isaaclab.sh -p` 而不是系統 Python。

### 3. CUDA 錯誤

```bash
# 檢查 CUDA 可用性
./isaaclab.sh -p -c "import torch; print(torch.cuda.is_available())"
```

### 4. 訓練很慢

- 使用 Headless 模式：`--headless`
- 減少環境數量：`--num_envs 16`
- 確保使用 GPU

**📖 更多問題排查請參閱：[指令快速參考](md/指令快速參考.md)**

---

## 📈 性能指標

### 訓練配置

- **並行環境數**：48
- **每環境步數**：24
- **總迭代次數**：3000
- **總訓練步數**：3,456,000
- **訓練時間**：60-90 分鐘（RTX 3090/4090）

### 預期結果

良好的訓練應該顯示：
- ✅ 平均獎勵 > -500（理想 > 0）
- ✅ 成功到達目標率 > 10%
- ✅ 超時率 < 80%
- ✅ 平均距離誤差 < 2.0m

---

## 📝 授權

本專案基於 Isaac Lab 的 BSD-3-Clause 授權。

---

## 🙏 致謝

- **NVIDIA Isaac Sim & Isaac Lab** - 提供強大的機器人模擬平台
- **RSL-RL** - 提供高效的 RL 訓練框架
- **Nova Carter** - NVIDIA 官方移動機器人平台

---

## 📞 支援

- **核心文檔**：
  - [訓練架構完整說明](md/訓練架構完整說明.md)
  - [指令快速參考](md/指令快速參考.md)

- **Isaac Lab 官方文檔**：https://isaac-sim.github.io/IsaacLab/
- **Isaac Sim 文檔**：https://docs.omniverse.nvidia.com/isaacsim/

---

**🎯 開始訓練您的 Nova Carter 導航 Agent！**

```bash
cd /home/aa/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-LocalPlanner-Carter-v0 \
    --num_envs 48 \
    --headless
```
