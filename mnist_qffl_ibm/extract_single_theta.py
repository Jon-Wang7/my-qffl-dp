import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# ========= 配置 =========
train_data_path = "../data/mnist/train_data.pkl"         # 替换为你的真实路径
output_json_path = "qtnorm_theta_100x1.json"
n_qubits = 3
n_fuzzy_mem = 2
num_samples_to_use = 100
# ========================

# 加载训练数据
train_data = torch.load(train_data_path, map_location="cpu")
print(type(train_data), train_data.shape)

if hasattr(train_data, "numpy"):
    train_data = train_data.numpy()

x_data = train_data[:num_samples_to_use, :n_qubits]  # 取前 100 条样本、前3个特征

# 模糊中心 m 和宽度 theta（可替换为你模型训练好的参数）
x_mean = np.mean(x_data, axis=0)
x_std = np.std(x_data, axis=0)
m = np.stack([x_mean - x_std * 0.5, x_mean + x_std * 0.5], axis=1)
theta = np.ones_like(m) * (x_std.reshape(-1, 1) * 0.8)

records = []

for idx, x in tqdm(enumerate(x_data), total=len(x_data)):
    fuzzy_list0 = np.zeros_like(x)
    fuzzy_list1 = np.zeros_like(x)
    for i in range(n_qubits):
        a = (-(x[i] - m[i, 0]) ** 2) / (2 * theta[i, 0] ** 2)
        b = (-(x[i] - m[i, 1]) ** 2) / (2 * theta[i, 1] ** 2)
        fuzzy_list0[i] = np.exp(a)
        fuzzy_list1[i] = np.exp(b)
    fuzzy_list = np.stack([fuzzy_list0, fuzzy_list1], axis=0)

    # 从 8 个组合中找出 μ 总和最大的那个
    best_theta = None
    best_combo = None
    best_score = -1

    for i in range(n_fuzzy_mem ** n_qubits):
        bin_str = format(i, f"0{n_qubits}b")
        mu_list = []
        for j in range(n_qubits):
            idx_fuzzy = int(bin_str[j])
            mu = fuzzy_list[idx_fuzzy, j]
            mu = np.clip(mu, 0, 1)
            mu_list.append(mu)
        mu_arr = np.array(mu_list)
        score = np.sum(mu_arr)
        if score > best_score:
            best_score = score
            best_theta = 2 * np.arcsin(np.sqrt(mu_arr))
            best_combo = bin_str

    records.append({
        "sample_idx": idx,
        "combination": best_combo,
        "theta": best_theta.tolist()
    })

# 保存 JSON
df = pd.DataFrame(records)
df.to_json(output_json_path, orient="records", lines=True)
print(f"✅ 共生成样本: {len(records)} 条")
print(f"✅ 已保存为: {output_json_path}")