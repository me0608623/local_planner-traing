#!/usr/bin/env python3
"""
Nova Carter 訓練日誌分析工具

自動分析訓練結果並提供診斷建議

⚠️ 使用方法:
    ./isaaclab.sh -p scripts/analyze_training_log.py
    ./isaaclab.sh -p scripts/analyze_training_log.py --file logs/rsl_rl/your_training.log
    ./isaaclab.sh -p scripts/analyze_training_log.py --stdin
    
    注意：必須使用 isaaclab.sh 而不是系統 python！
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class TrainingAnalyzer:
    """訓練日誌分析器"""
    
    def __init__(self, log_text: str):
        self.log_text = log_text
        self.metrics = self.extract_metrics()
        
    def extract_metrics(self) -> Dict[str, float]:
        """從日誌中提取關鍵指標"""
        metrics = {}
        
        patterns = {
            'mean_reward': r'Mean reward:\s*([-\d.]+)',
            'episode_length': r'Mean episode length:\s*([\d.]+)',
            'reached_goal': r'Episode_Reward/reached_goal:\s*([\d.]+)',
            'progress_to_goal': r'Episode_Reward/progress_to_goal:\s*([-\d.]+)',
            'collision_penalty': r'Episode_Reward/collision_penalty:\s*([-\d.]+)',
            'obstacle_penalty': r'Episode_Reward/obstacle_proximity_penalty:\s*([-\d.]+)',
            'time_out_rate': r'Episode_Termination/time_out:\s*([\d.]+)',
            'goal_reached_rate': r'Episode_Termination/goal_reached:\s*([\d.]+)',
            'collision_rate': r'Episode_Termination/collision:\s*([\d.]+)',
            'position_error': r'position_error:\s*([\d.]+)',
            'value_loss': r'Mean value_function loss:\s*([\d.]+)',
            'entropy_loss': r'Mean entropy loss:\s*([\d.]+)',
        }
        
        for name, pattern in patterns.items():
            match = re.search(pattern, self.log_text)
            if match:
                metrics[name] = float(match.group(1))
            else:
                metrics[name] = 0.0
                
        return metrics
    
    def diagnose(self) -> Tuple[str, List[str], List[str]]:
        """診斷訓練狀態並提供建議"""
        issues = []
        suggestions = []
        
        m = self.metrics
        
        # 整體狀態評估
        if m['mean_reward'] > 0:
            status = "✅ 良好"
        elif m['mean_reward'] > -500:
            status = "⚠️ 需要改進"
        else:
            status = "❌ 表現不佳"
        
        # 問題診斷
        if m['mean_reward'] < -1000:
            issues.append("❌ 平均獎勵過低 ({:.2f})".format(m['mean_reward']))
            suggestions.append("調整獎勵函數權重，增加正向引導獎勵")
        
        if m['goal_reached_rate'] == 0.0:
            issues.append("❌ 從未成功到達目標")
            suggestions.append("簡化環境：減少障礙物數量，縮短目標距離")
            suggestions.append("增加 episode 時間限制")
        elif m['goal_reached_rate'] < 0.1:
            issues.append("⚠️ 成功率過低 ({:.1%})".format(m['goal_reached_rate']))
            suggestions.append("調整獎勵函數以提供更好的引導")
        
        if m['time_out_rate'] > 0.9:
            issues.append("❌ 幾乎所有 episode 都超時 ({:.1%})".format(m['time_out_rate']))
            suggestions.append("增加 episode_length_s 配置")
            suggestions.append("簡化任務難度")
        
        if m['position_error'] > 4.0:
            issues.append("❌ 距離目標太遠 ({:.2f}m)".format(m['position_error']))
            suggestions.append("檢查觀測空間是否包含目標位置信息")
            suggestions.append("增加朝向目標的獎勵引導")
        
        if m['collision_rate'] > 0.5:
            issues.append("⚠️ 碰撞率過高 ({:.1%})".format(m['collision_rate']))
            suggestions.append("增加避障獎勵權重")
            suggestions.append("檢查 LiDAR 觀測是否正確")
        
        if m['value_loss'] > 1000:
            issues.append("⚠️ Value function loss 過高 ({:.2f})".format(m['value_loss']))
            suggestions.append("降低學習率")
            suggestions.append("檢查獎勵尺度是否過大")
        
        if m['progress_to_goal'] < -50:
            issues.append("❌ 沒有接近目標的行為 ({:.2f})".format(m['progress_to_goal']))
            suggestions.append("增加 progress_to_goal_weight")
            suggestions.append("減少其他懲罰權重")
        
        # 正面反饋
        if not issues:
            issues.append("✅ 訓練狀態良好！")
            suggestions.append("繼續訓練以進一步提升性能")
        
        return status, issues, suggestions
    
    def print_analysis(self):
        """打印詳細分析報告"""
        print("=" * 80)
        print("📊 Nova Carter 訓練結果分析")
        print("=" * 80)
        
        # 關鍵指標
        print("\n📈 關鍵指標:")
        print(f"   平均獎勵: {self.metrics['mean_reward']:.2f}")
        print(f"   成功率: {self.metrics['goal_reached_rate']:.1%}")
        print(f"   超時率: {self.metrics['time_out_rate']:.1%}")
        print(f"   碰撞率: {self.metrics['collision_rate']:.1%}")
        print(f"   平均距離誤差: {self.metrics['position_error']:.2f}m")
        print(f"   平均 episode 長度: {self.metrics['episode_length']:.2f}")
        
        # 獎勵分解
        print("\n💰 獎勵分解:")
        print(f"   到達目標獎勵: {self.metrics['reached_goal']:.2f}")
        print(f"   接近目標獎勵: {self.metrics['progress_to_goal']:.2f}")
        print(f"   碰撞懲罰: {self.metrics['collision_penalty']:.2f}")
        print(f"   障礙物接近懲罰: {self.metrics['obstacle_penalty']:.2f}")
        
        # 訓練指標
        print("\n🔧 訓練指標:")
        print(f"   Value function loss: {self.metrics['value_loss']:.2f}")
        print(f"   Entropy loss: {self.metrics['entropy_loss']:.2f}")
        
        # 診斷結果
        status, issues, suggestions = self.diagnose()
        
        print(f"\n🎯 整體狀態: {status}")
        
        print("\n🔍 發現的問題:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n💡 改進建議:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        # 配置建議
        print("\n⚙️ 建議的配置調整:")
        if self.metrics['goal_reached_rate'] == 0.0:
            print("   # 簡化環境配置")
            print("   scene.obstacles.num_obstacles = 3  # 減少障礙物")
            print("   commands.goal_command.ranges.distance = (2.0, 4.0)  # 縮短距離")
            print("   episode_length_s = 30.0  # 增加時間")
        
        if self.metrics['mean_reward'] < -1000:
            print("   # 調整獎勵權重")
            print("   progress_to_goal_weight = 5.0  # 增加引導")
            print("   collision_penalty_weight = -5.0  # 減少懲罰")
        
        if self.metrics['value_loss'] > 1000:
            print("   # 調整訓練超參數")
            print("   learning_rate = 3e-4  # 降低學習率")
            print("   num_steps_per_env = 24  # 增加步數")
        
        print("\n" + "=" * 80)


def analyze_log_file(log_file: Path):
    """分析日誌文件"""
    with open(log_file, 'r') as f:
        log_text = f.read()
    
    # 找到最後一次迭代的日誌
    iterations = re.findall(
        r'Learning iteration \d+/\d+.*?(?=Learning iteration|\Z)',
        log_text,
        re.DOTALL
    )
    
    if not iterations:
        print("❌ 無法在日誌文件中找到訓練迭代信息")
        return
    
    # 分析最後一次迭代
    last_iteration = iterations[-1]
    analyzer = TrainingAnalyzer(last_iteration)
    analyzer.print_analysis()


def analyze_text(text: str):
    """分析文本輸入"""
    analyzer = TrainingAnalyzer(text)
    analyzer.print_analysis()


def main():
    parser = argparse.ArgumentParser(
        description="Nova Carter 訓練日誌分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析日誌文件
  python analyze_training_log.py --file logs/rsl_rl/local_planner/output.log
  
  # 從剪貼板分析（將訓練輸出粘貼後按 Ctrl+D）
  python analyze_training_log.py --stdin
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=Path,
        help='訓練日誌文件路徑'
    )
    
    parser.add_argument(
        '--stdin',
        action='store_true',
        help='從標準輸入讀取日誌'
    )
    
    args = parser.parse_args()
    
    if args.file:
        if not args.file.exists():
            print(f"❌ 文件不存在: {args.file}")
            return
        analyze_log_file(args.file)
    elif args.stdin:
        print("📋 請粘貼訓練日誌（完成後按 Ctrl+D）:")
        import sys
        text = sys.stdin.read()
        analyze_text(text)
    else:
        # 分析示例日誌
        example_log = """
                      Learning iteration 999/1000                       

                       Computation: 20 steps/s (collection: 1.634s, learning 0.656s)
             Mean action noise std: 1.07
          Mean value_function loss: 3154.6131
               Mean surrogate loss: 0.0016
                 Mean entropy loss: 2.9716
                       Mean reward: -2598.61
               Mean episode length: 181.36
   Episode_Reward/progress_to_goal: -125.6530
       Episode_Reward/reached_goal: 0.0000
Episode_Reward/obstacle_proximity_penalty: 0.0000
  Episode_Reward/collision_penalty: 0.0000
    Episode_Reward/ang_vel_penalty: 0.3417
 Episode_Reward/standstill_penalty: 0.0002
Metrics/goal_command/position_error: 5.1919
Metrics/goal_command/orientation_error: 0.0000
      Episode_Termination/time_out: 1.0000
  Episode_Termination/goal_reached: 0.0000
     Episode_Termination/collision: 0.0000
        """
        print("使用示例日誌進行分析...\n")
        analyze_text(example_log)
        print("\n提示: 使用 --file 或 --stdin 參數分析實際日誌")


if __name__ == "__main__":
    main()
