import re
from collections import defaultdict

log_file = "robust_out/logs.log"   # 修改为你的log路径

# 匹配 averaged stats 行
pattern = re.compile(
    r"Averaged stats: pixel-level F1: \[local: ([0-9.]+) \| reduced: ([0-9.]+)\]"
)

datasets = ["coverage", "casia_v1", "columbia", "nist2016"]

# 保存结构：
# results[wrapper_name][param][dataset] = reduced_f1
results = defaultdict(lambda: defaultdict(dict))

current_wrapper = None
current_param = None
dataset_idx = 0

with open(log_file, "r") as f:
    for line in f:
        # 识别增强方法和参数
        if "[ROBUST TEST]" in line:
            # line 例如： [ROBUST TEST] RotationWrapper param=0.35
            parts = line.strip().split()
            wrapper = parts[2]     # RotationWrapper
            param = parts[3]       # param=0.35
            current_wrapper = wrapper
            current_param = param
            dataset_idx = 0
        
        # 匹配 averaged 结果
        match = pattern.search(line)
        if match and current_wrapper is not None:
            reduced = float(match.group(2))
            dataset_name = datasets[dataset_idx]
            results[current_wrapper][current_param][dataset_name] = reduced
            dataset_idx += 1

# === 输出 & 计算 ===
print("\n========== Robustness Summary ==========\n")
for wrapper, params in results.items():
    print(f"\n### {wrapper} ###")
    
    # 1) 输出每个参数每个数据集及其均值
    for param, vals in params.items():
        vals_list = list(vals.values())
        mean_val = sum(vals_list) / len(vals_list)
        print(f"  {param}: " +
              ", ".join(f"{ds}={vals[ds]:.4f}" for ds in datasets) +
              f"  -> Mean={mean_val:.4f}")
    
    # 2) 计算剔除 param=0 的总体平均
    valid_params = [vals for p, vals in params.items() if p != "param=0"]
    if valid_params:
        avg_across = sum(sum(v.values()) / len(v.values()) for v in valid_params) / len(valid_params)
        print(f"\n  => Final Robustness Score (exclude param=0): {avg_across:.4f}")
