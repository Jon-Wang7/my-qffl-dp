import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# === Load files ===
test_label = torch.load('../data/mnist/test_label.pkl')[:2000]
lambda_weights = torch.load('../result/data/mni_qffl/lambda_weights.pt', weights_only=False)  # shape: [batch, node]
output_probs = torch.load('../result/data/mni_qffl/output_probs.pt', weights_only=False)  # shape: [batch, class]

# === Step 1: Simulate member / non-member ===
batch_size = len(test_label)
is_member = np.array([1 if i < batch_size // 2 else 0 for i in range(batch_size)])

# === Step 2: Calculate λ entropy ===
lambda_entropy = -torch.sum(lambda_weights * torch.log(lambda_weights + 1e-8), dim=1).cpu().numpy()

# === Step 3: Calculate λ max value ===
lambda_max = torch.max(lambda_weights, dim=1).values.cpu().numpy()

# === Step 4: Calculate output softmax confidence and entropy ===
output_max_prob = torch.max(output_probs, dim=1).values.cpu().numpy()
output_entropy = -torch.sum(output_probs * torch.log(output_probs + 1e-8), dim=1).cpu().numpy()

print(len(is_member), len(lambda_entropy), len(lambda_max), len(output_max_prob), len(output_entropy))

# === Step 5: Build DataFrame for analysis ===
df = pd.DataFrame({
    'is_member': is_member,
    'lambda_entropy': lambda_entropy,
    'lambda_max': lambda_max,
    'output_max_prob': output_max_prob,
    'output_entropy': output_entropy
})

df.to_csv("privacy_indicator_analysis.csv", index=False)
print("✅ 数据已保存为 privacy_indicator_analysis.csv")

# === Step 6: Visualize distributions ===
plt.figure(figsize=(12, 8))
for i, col in enumerate(['lambda_entropy', 'lambda_max', 'output_max_prob', 'output_entropy']):
    plt.subplot(2, 2, i + 1)
    sns.kdeplot(df[df['is_member'] == 1][col], label='member', fill=True)
    sns.kdeplot(df[df['is_member'] == 0][col], label='non-member', fill=True)
    plt.title(f'{col} Distribution')
    plt.legend()
plt.tight_layout()
plt.savefig("privacy_distributions.png")
plt.show()

# === Step 7: Simple AUC Evaluation ===
for col in ['lambda_entropy', 'lambda_max', 'output_max_prob', 'output_entropy']:
    auc = roc_auc_score(df['is_member'], df[col])
    print(f"AUC({col}): {auc:.4f}")
