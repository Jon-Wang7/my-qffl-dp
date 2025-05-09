import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# 参数空间
noise_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
clip_list = [1, 5, 10, 20]


def run_evaluation(noise, clip):
    # === Load files ===
    test_label = torch.load('../data/mnist/test_label.pkl')[:2000]
    lambda_weights = torch.load(f'../result/data/mni_qffl_dp/noise_{noise}_clip_{clip}_output_probs.pt',
                                weights_only=False)  # shape: [batch, node]
    output_probs = torch.load(f'../result/data/mni_qffl_dp/noise_{noise}_clip_{clip}_output_probs.pt',
                              weights_only=False)  # shape: [batch, class]
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

    # print(f'noise_{noise}_clip_{clip} df : {df}')
    # === Step 6: Visualize distributions ===
    # plt.figure(figsize=(12, 8))
    # for i, col in enumerate(['lambda_entropy', 'lambda_max', 'output_max_prob', 'output_entropy']):
    #     plt.subplot(2, 2, i + 1)
    #     sns.kdeplot(df[df['is_member'] == 1][col], label='member', fill=True)
    #     sns.kdeplot(df[df['is_member'] == 0][col], label='non-member', fill=True)
    #     plt.title(f'{col} Distribution')
    #     plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"noise_{noise}_clip_{clip}_privacy_distributions.png")
    # plt.show()
    # === Step 7: Simple AUC Evaluation ===
    for col in ['lambda_entropy', 'lambda_max', 'output_max_prob', 'output_entropy']:
        auc = roc_auc_score(df['is_member'], df[col])
        print(f"AUC({col}): {auc:.4f}")


if __name__ == '__main__':
    for noise in noise_list:
        for clip in clip_list:
            print(f'run noise_{noise}_clip_{clip}')
            run_evaluation(noise, clip)
