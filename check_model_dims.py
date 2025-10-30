#!/usr/bin/env python3
"""檢查模型的實際輸入輸出維度"""
import torch
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/rsl_rl/local_planner_carter/2025-10-27_17-29-26/model_4999.pt'

checkpoint = torch.load(model_path, map_location='cpu')

print("=" * 80)
print("模型維度檢查")
print("=" * 80)

# Actor第一層
first_layer_weight = checkpoint['model_state_dict']['actor.0.weight']
print(f"Actor第一層權重shape: {first_layer_weight.shape}")
print(f"→ 輸入維度（期望的觀測維度）: {first_layer_weight.shape[1]}")

# Actor最後一層
actor_keys = [k for k in checkpoint['model_state_dict'].keys() if 'actor' in k]
last_layer = [k for k in actor_keys if 'weight' in k][-1]
last_layer_weight = checkpoint['model_state_dict'][last_layer]
print(f"Actor最後一層權重shape: {last_layer_weight.shape}")
print(f"→ 輸出維度（動作維度）: {last_layer_weight.shape[0]}")

print("=" * 80)


