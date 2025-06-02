import torch
from tqdm import tqdm
from mni_QFNN_DP import Qfnn
from common.utils import setup_seed
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 参数空间
# noise_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
# clip_list = [1, 5, 10, 20]

noise_list = [0.1]
clip_list = [5]

DEVICE = torch.device('cpu')
NAME = 'mnist_QFNN_gas_q4_star'
setup_seed(777)
node = 5
# #测试
test_data = torch.load('../data/mnist/test_data.pkl').to(DEVICE)[:2000]
label = torch.load('../data/mnist/test_label.pkl').to(DEVICE)[:2000]


def run_test(noise, clip):
    global label
    gmm_list = torch.load(f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{clip}_mnist_QFNN_gas_q4_star_gmm_list',
                          weights_only=False)
    data_weights = torch.load(
        f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{clip}_mnist_QFNN_gas_q4_star_data_weights')
    gmm_scores = []
    for i in range(node):
        gmm_scores.append(gmm_list[i].score_samples(test_data.cpu().numpy()))
    gmm_scores = torch.tensor(np.array(gmm_scores)).to(DEVICE).permute(1, 0)
    for i in range(node):
        gmm_scores[:, i] = gmm_scores[:, i] * data_weights[i]
    sum = torch.sum(gmm_scores, dim=1)
    for i in range(node):
        gmm_scores[:, i] = gmm_scores[:, i] / sum
    out_put = []
    for i in tqdm(range(node)):
        model = Qfnn(DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(f'../result/model/mni_qffl_dp_cl/last_noise_{noise}_clip_{clip}_{NAME}_n{i}.pth'))
        model.eval()
        with torch.no_grad():
            out_put.append(model(test_data))
    out_put = torch.stack(out_put, dim=1)
    out_put = torch.softmax(out_put, dim=1)
    for i in range(node):
        m = out_put[:, i, :]
        n = gmm_scores[:, i].unsqueeze(1)
        out_put[:, i, :] = out_put[:, i, :] * gmm_scores[:, i].unsqueeze(1)
    output = torch.sum(out_put, dim=1)
    pred = torch.argmin(output, dim=1)
    pred = pred.cpu().numpy()
    label = np.asarray(label)
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')
    print(f'noise_{noise}_clip_{clip} acc:{acc} precision:{precision} recall:{recall} f1:{f1}')
    cm = confusion_matrix(label, pred)
    # === 保存预测结果和融合权重（用于隐私评估） ===
    torch.save(output, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{clip}_output_probs.pt')  # [batch, class]
    torch.save(gmm_scores, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{clip}_lambda_weights.pt')  # [batch, node]
    print(f"noise_{noise}_clip_{clip}预测输出和 GMM 权重已保存！")


if __name__ == '__main__':
    for noise in noise_list:
        for clip in clip_list:
            print(f'run noise_{noise}_clip_{clip}')
            run_test(noise, clip)
