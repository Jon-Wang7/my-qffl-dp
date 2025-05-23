# generate_qtnorm_theta.py

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# ========= 可配置参数 =========
train_data_path = "../data/mnist/train_data.pkl"   # 替换为你实际的路径
output_json_path = "qtnorm_ibm_upload_preview.json"
n_qubits = 3
n_fuzzy_mem = 2
num_samples_to_use = None  # 设置为整数如 1000 或 None 表示使用全部
# ============================

train_data = torch.load(train_data_path, map_location="cpu")
print(type(train_data), train_data.shape)

if hasattr(train_data, "numpy"):
    train_data = train_data.numpy()

if num_samples_to_use is not None:
    train_data = train_data[:num_samples_to_use]

x_data = train_data[:, :n_qubits]  # 每个样本只取前3个特征

# 初始化模糊参数（可替换为模型参数）
np.random.seed(42)
m = np.random.rand(n_qubits, n_fuzzy_mem)
theta = np.random.rand(n_qubits, n_fuzzy_mem) * 0.5

records = []

print(f"开始处理 {len(x_data)} 个样本，每个样本生成 {n_fuzzy_mem**n_qubits} 个量子输入...")

for idx, x in tqdm(enumerate(x_data), total=len(x_data)):
    fuzzy_list0 = np.zeros_like(x)
    fuzzy_list1 = np.zeros_like(x)
    for i in range(n_qubits):
        a = (-(x[i] - m[i, 0]) ** 2) / (2 * theta[i, 0] ** 2)
        b = (-(x[i] - m[i, 1]) ** 2) / (2 * theta[i, 1] ** 2)
        fuzzy_list0[i] = np.exp(a)
        fuzzy_list1[i] = np.exp(b)
    fuzzy_list = np.stack([fuzzy_list0, fuzzy_list1], axis=0)

    for i in range(n_fuzzy_mem ** n_qubits):
        bin_str = format(i, f"0{n_qubits}b")
        mu_list = []
        for j in range(n_qubits):
            idx_fuzzy = int(bin_str[j])
            mu = fuzzy_list[idx_fuzzy, j]
            mu = np.clip(mu, 0, 1)
            mu_list.append(mu)
        mu_arr = np.array(mu_list)
        theta_arr = 2 * np.arcsin(np.sqrt(mu_arr))
        records.append({
            "sample_idx": idx,
            "combination": bin_str,
            "theta": theta_arr.tolist()
        })

df = pd.DataFrame(records)
df.to_json(output_json_path, orient="records", lines=True)

print(f"✅ 总样本数: {len(x_data)}")
print(f"✅ 总量子输入组合数: {len(records)}")
print(f"✅ 已保存为: {output_json_path}")