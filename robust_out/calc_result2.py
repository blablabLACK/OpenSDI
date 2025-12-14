import re
from collections import defaultdict

def parse_log_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    wrapper_name = None
    data = defaultdict(lambda: defaultdict(list))  
    # data[wrapper][metric] = list of values across params

    for line in lines:
        # 解析 Wrapper 名字
        m = re.match(r"###\s*(\w+)\s*###", line)
        if m:
            wrapper_name = m.group(1)
            continue
        
        # 匹配包含指标的行
        pattern = (r"coverage=([0-9.]+),\s*casia_v1=([0-9.]+),\s*"
                   r"columbia=([0-9.]+),\s*nist2016=([0-9.]+)")
        m = re.search(pattern, line)
        if m and wrapper_name:
            cov, casia, col, nist = map(float, m.groups())
            data[wrapper_name]["coverage"].append(cov)
            data[wrapper_name]["casia_v1"].append(casia)
            data[wrapper_name]["columbia"].append(col)
            data[wrapper_name]["nist2016"].append(nist)

    return data


def compute_and_print(data):
    print("\n========== Average Results Per Wrapper ==========\n")
    for wrapper, metrics in data.items():
        print(f"### {wrapper} ###")
        for metric_name, values in metrics.items():
            avg = sum(values) / len(values)
            print(f"  {metric_name}: {avg:.4f}")
        print()


if __name__ == "__main__":
    log_path = "out_robust_result.log"   # ← ← 修改为你的 log 文件路径
    data = parse_log_file(log_path)
    compute_and_print(data)
