import torch
from tqdm import tqdm
from mni_QFNN import Qfnn
from common.utils import setup_seed
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

DEVICE = torch.device('cpu')
NAME = 'mnist_QFNN_gas_q4_star'
setup_seed(777)
node = 9

# 加载测试集（前2000条）
test_data = torch.load('../data/mnist/test_data.pkl').to(DEVICE)[:2000]
label = torch.load('../data/mnist/test_label.pkl').to(DEVICE)[:2000]

# 加载 GMM 和数据权重
gmm_list = torch.load(f'../result/data/mni_qffl/{NAME}_gmm_list', weights_only=False)
data_weights = torch.tensor(torch.load(f'../result/data/mni_qffl/{NAME}_data_weights')).to(DEVICE)

# === Step 1: 获取 GMM score_samples 输出（已为 log density，不要再 log） ===
gmm_scores = []
for i in range(node):
    gmm_scores.append(gmm_list[i].score_samples(test_data.cpu().numpy()))

# 转换成 tensor，shape: [batch, node]
log_scores = torch.tensor(np.array(gmm_scores)).to(DEVICE).permute(1, 0)
log_scores = torch.clamp(log_scores, min=-50)  # 防止极小值导致数值爆炸
log_weights = torch.log(data_weights + 1e-8)

# λ_h = softmax(log(score) + log(weight))
temperature = 1  # 可设为 0.3 - 2.0
fusion_logits = log_scores + log_weights.unsqueeze(0)  # broadcast
lambda_weights = F.softmax(fusion_logits / temperature, dim=1)

# === Step 2: 每个 node 推理输出 y_h ===
out_put = []
for i in tqdm(range(node)):
    model = Qfnn(DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(f'../result/model/mni_qffl/{NAME}_n{i}.pth'))
    model.eval()
    with torch.no_grad():
        out_put.append(model(test_data))

# shape: [batch, node, class]
out_put = torch.stack(out_put, dim=1)
out_put = torch.softmax(out_put, dim=2)

# === Step 3: 融合输出 ===
y_weighted = out_put * lambda_weights.unsqueeze(2)  # 广播乘法
output = torch.sum(y_weighted, dim=1)

# === Step 4: 评估指标 ===
pred = torch.argmax(output, dim=1).cpu().numpy()
label = label.cpu().numpy()
acc = accuracy_score(label, pred)
precision = precision_score(label, pred, average='macro', zero_division=0)
recall = recall_score(label, pred, average='macro', zero_division=0)
f1 = f1_score(label, pred, average='macro', zero_division=0)
print(f'acc: {acc:.4f}  precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}')

# 可选混淆矩阵
cm = confusion_matrix(label, pred)

# === 保存预测结果和融合权重（用于隐私评估） ===
torch.save(output, '../result/data/mni_qffl/output_probs.pt')  # ✅ 正确保存预测输出
torch.save(lambda_weights, '../result/data/mni_qffl/lambda_weights.pt')  # ✅ 正确保存 softmax 后的 λ 权重

print("预测输出和 GMM 权重已保存！")

# === 可视化 λ 分布情况（Debug 推荐）===
torch.set_printoptions(precision=4, sci_mode=False)
print("示例 λ 权重 (前5条样本):", lambda_weights[:5])
print("各 client 的 λ 平均值:", torch.mean(lambda_weights, dim=0))
