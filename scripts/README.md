# 🔧 腳本工具集

> **Nova Carter 訓練專案** - 所有腳本工具的使用說明

---

## 🚀 快速開始

### 統一啟動入口（推薦）

```bash
cd /home/aa/IsaacLab
./train.sh
```

**功能**：
- ✅ 選單式選擇訓練模式
- ✅ 無頭模式 / GUI 模式
- ✅ 監控現有訓練

---

## 📂 腳本分類

### 📁 training/ - 訓練腳本

| 腳本 | 說明 | 使用場景 |
|------|------|----------|
| `start_v4_training.sh` | v4 無頭訓練（最快） | 正式訓練 |
| `start_v4_training_gui.sh` | v4 GUI 訓練 | 觀察訓練過程 |
| `啟動24環境訓練.sh` | 24 環境並行訓練（GUI） | 可視化 24 個機器人 |
| `train_debug_correct.sh` | 調試訓練 | 測試配置 |

---

### 📁 monitoring/ - 監控腳本

| 腳本 | 說明 | 使用方式 |
|------|------|----------|
| `monitor_training.sh` | 即時監控訓練狀態 | `./scripts/monitoring/monitor_training.sh` |

**功能**：
- ✅ 顯示訓練進程狀態
- ✅ GPU 使用率
- ✅ 最新訓練數據
- ✅ 關鍵指標（Progress, Standstill, Position Error）

**使用**：
```bash
# 單次查看
./scripts/monitoring/monitor_training.sh

# 持續監控（每 5 秒刷新）
watch -n 5 ./scripts/monitoring/monitor_training.sh
```

---

### 📁 setup/ - 環境設置腳本

| 腳本 | 說明 | 使用場景 |
|------|------|----------|
| `setup_env_fix_pinocchio.sh` | 修復 Pinocchio 問題 | 環境錯誤時 |
| `remove_pinocchio.sh` | 移除 Pinocchio | 清理環境 |
| `verify_local_usd.sh` | 驗證 USD 模型 | 檢查資源 |

---

## 🎯 常用任務

### 1️⃣ 啟動訓練

**最簡單方式**：
```bash
cd /home/aa/IsaacLab
./train.sh
# 選擇 1（無頭模式）或 2（GUI 模式）
```

**直接啟動無頭模式**：
```bash
./scripts/training/start_v4_training.sh
```

**直接啟動 GUI 模式**：
```bash
./scripts/training/啟動24環境訓練.sh
```

---

### 2️⃣ 監控訓練

**即時查看**：
```bash
./scripts/monitoring/monitor_training.sh
```

**持續監控**：
```bash
watch -n 5 ./scripts/monitoring/monitor_training.sh
```

**查看詳細日誌**：
```bash
tail -f training_v4.log
```

---

### 3️⃣ 修復環境

**Pinocchio 錯誤**：
```bash
./scripts/setup/setup_env_fix_pinocchio.sh
```

**驗證 USD 模型**：
```bash
./scripts/setup/verify_local_usd.sh
```

---

## 📊 訓練模式對比

| 模式 | 腳本 | 速度 | GPU 使用 | 視窗 | 適用場景 |
|------|------|------|----------|------|----------|
| **無頭模式** | `start_v4_training.sh` | 最快 ⚡ | 低 | ❌ | 正式訓練 |
| **GUI 模式** | `啟動24環境訓練.sh` | 較慢 🐢 | 高 | ✅ | 觀察學習 |
| **調試模式** | `train_debug_correct.sh` | 中等 | 中 | 可選 | 測試配置 |

---

## 🔧 腳本詳細說明

### start_v4_training.sh（無頭模式）

**功能**：
- 啟動 v4 配置訓練
- 無 GUI 視窗（headless）
- 使用 WandB 記錄
- 訓練速度最快

**參數**：
- 任務：`Isaac-Navigation-LocalPlanner-Min-v0`
- 環境數：24
- 迭代數：10000
- 模式：headless

**使用**：
```bash
cd /home/aa/IsaacLab
./scripts/training/start_v4_training.sh
```

---

### 啟動24環境訓練.sh（GUI 模式）

**功能**：
- 啟動 v4 配置訓練
- 顯示 Isaac Sim GUI 視窗
- 可看到 24 個機器人並排訓練
- 訓練速度較慢但直觀

**GUI 操作**：
- 按 `A` + `F` 鍵：聚焦全場景
- 滑鼠滾輪：縮放視角
- 滑鼠左鍵：旋轉視角

**使用**：
```bash
cd /home/aa/IsaacLab
./scripts/training/啟動24環境訓練.sh
```

---

### monitor_training.sh（監控腳本）

**功能**：
- 檢查訓練進程狀態
- 顯示 GPU 使用率
- 提取最新訓練數據
- 顯示關鍵指標

**輸出內容**：
```
✅ 訓練進程運行中（PID: xxxxx）
✅ Isaac Sim 進程數：3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎮 GPU 狀態：
  GPU 0: NVIDIA GeForce RTX 5090
  使用率: 94% | 記憶體: 18140/32607 MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 最新訓練數據：
  Learning iteration 1000/10000
  Mean Reward: -15.32
  Progress: 0.0523      ← 關鍵！應該 > 0
  Standstill: -0.2134   ← 應該 > -0.5
  Position Error: 2.87  ← 應該 < 3.0m
```

**使用**：
```bash
# 單次查看
./scripts/monitoring/monitor_training.sh

# 持續監控（推薦）
watch -n 5 ./scripts/monitoring/monitor_training.sh
```

---

### setup_env_fix_pinocchio.sh（環境修復）

**功能**：
- 修復 Pinocchio 依賴問題
- 重新配置環境變數
- 解決常見環境錯誤

**何時使用**：
- 出現 Pinocchio 相關錯誤
- 訓練啟動失敗
- 環境配置問題

**使用**：
```bash
./scripts/setup/setup_env_fix_pinocchio.sh
source ~/.bashrc
```

---

## 🌐 其他監控方式

### WandB 雲端監控（推薦）

**訪問**：https://wandb.ai/

**優點**：
- ✅ 漂亮的曲線圖
- ✅ 即時更新
- ✅ 可多裝置查看
- ✅ 歷史記錄完整

**專案**：`nova-carter-navigation`

---

### TensorBoard 本地監控

```bash
cd /home/aa/IsaacLab
tensorboard --logdir logs/rsl_rl/local_planner_carter

# 瀏覽器訪問：http://localhost:6006
```

---

## 💡 最佳實踐

### 正式訓練流程

```bash
# 1. 啟動無頭訓練（最快）
./scripts/training/start_v4_training.sh

# 2. 另開終端持續監控
watch -n 5 ./scripts/monitoring/monitor_training.sh

# 3. 打開 WandB 查看曲線
# https://wandb.ai/

# 4. 30 分鐘後檢查 Progress 是否回正
./scripts/monitoring/monitor_training.sh
```

### 首次觀察訓練

```bash
# 1. 啟動 GUI 模式
./scripts/training/啟動24環境訓練.sh

# 2. GUI 出現後：
#    - 按 A（選擇全部）
#    - 按 F（聚焦全場景）
#    - 觀察 24 個機器人

# 3. 確認學習行為正常後
#    - 停止 GUI 訓練（Ctrl+C）
#    - 改用無頭模式繼續
```

---

## 🐛 疑難排解

### 訓練無法啟動

```bash
# 1. 檢查環境
./scripts/setup/verify_local_usd.sh

# 2. 修復 Pinocchio
./scripts/setup/setup_env_fix_pinocchio.sh

# 3. 清理殘留進程
pkill -f train.py
pkill -f isaac

# 4. 重新啟動
./scripts/training/start_v4_training.sh
```

### 監控腳本無數據

```bash
# 原因：訓練剛啟動，日誌尚未生成
# 解決：等待 1-2 分鐘後重試

./scripts/monitoring/monitor_training.sh
```

### GUI 視窗未出現

```bash
# 1. 檢查進程是否運行
ps aux | grep train.py

# 2. 等待 2-4 分鐘（初始化需要時間）

# 3. 如果仍未出現
#    - 切換到無頭模式（更快）
#    - 使用 WandB 監控即可
```

---

## 📞 快速指令參考

```bash
# 啟動訓練
./train.sh                                      # 選單啟動
./scripts/training/start_v4_training.sh         # 無頭模式
./scripts/training/啟動24環境訓練.sh            # GUI 模式

# 監控訓練
./scripts/monitoring/monitor_training.sh        # 單次查看
watch -n 5 ./scripts/monitoring/monitor_training.sh  # 持續監控
tail -f training_v4.log                         # 詳細日誌

# 停止訓練
pkill -f train.py                               # 停止訓練進程

# 環境修復
./scripts/setup/setup_env_fix_pinocchio.sh     # 修復 Pinocchio
./scripts/setup/verify_local_usd.sh            # 驗證資源

# WandB 監控
https://wandb.ai/                               # 雲端監控
```

---

## 🔄 腳本更新記錄

**2025-10-31**：
- ✅ 整理腳本到分類資料夾
- ✅ 創建統一啟動入口 `train.sh`
- ✅ 新增監控腳本 `monitor_training.sh`

---

**從 `train.sh` 開始，一鍵啟動訓練！** 🚀

