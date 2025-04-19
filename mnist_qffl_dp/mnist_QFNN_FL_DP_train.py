import torch
import torch.nn as nn
from mni_QFNN_DP import Qfnn
import numpy as np
from common.utils import acc_cal, setup_seed
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from pyvacy import optim as pyvacy_optim
from pyvacy.analysis import moments_accountant
from tqdm import tqdm
import json
import os
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity

BATCH_SIZE = 128
EPOCH = 10
LR = 0.05
# noise_multiplier = 1.0
# l2_norm_clip = 10
DEVICE = torch.device('cpu')
PATIENCE = 5  # 增加耐心值，避免过早停止
MIN_DELTA = 0.0005  # 减小最小提升阈值
MIN_EPOCHS = 3  # 最小训练轮数

setup_seed(777)

transform = transforms.Compose([transforms.ToTensor()])

# 加载训练集和测试集
train_data = torch.load('../data/mnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/mnist/train_label.pkl').cpu().numpy()
test_data = torch.load('../data/mnist/test_data.pkl').cpu().numpy()
test_label = torch.load('../data/mnist/test_label.pkl').cpu().numpy()

all_len = len(train_label)

gmm_list = []
weights = []

NAME = f'mnist_QFNN_gas_q4_star'
node = 9

noise_multipliers = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
l2_norm_clips = [1, 5, 10, 20]


# 修改EarlyStopping类
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA, mode='max', min_epochs=MIN_EPOCHS):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.epoch = 0
        
    def __call__(self, value):
        self.epoch += 1
        if self.epoch < self.min_epochs:
            return False
            
        if self.best_value is None:
            self.best_value = value
        elif (self.mode == 'max' and value <= self.best_value + self.min_delta) or \
             (self.mode == 'min' and value >= self.best_value - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_value = value
            self.counter = 0
            
        return self.early_stop


keep_list = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]]

# 在循环开始前添加结果字典
results = {
    'noise_multipliers': noise_multipliers,
    'l2_norm_clips': l2_norm_clips,
    'node_results': {}
}

for noise in noise_multipliers:
    for l2_norm in l2_norm_clips:
        node_results = {
            'accuracies': [],
            'epsilons': [],
            'final_acc': 0,
            'avg_epsilon': 0,
            'test_accuracies': [],  # 添加测试集准确率记录
            'final_test_acc': 0  # 添加最终测试集准确率
        }

        for i in range(node):
            model = Qfnn(DEVICE).to(DEVICE)
            optimizer = pyvacy_optim.DPSGD(
                params=model.parameters(),
                lr=LR,
                batch_size=BATCH_SIZE,
                noise_multiplier=noise,
                l2_norm_clip=l2_norm
            )
            loss_func = nn.CrossEntropyLoss()
            early_stopping = EarlyStopping(mode='max')  # 使用准确率作为指标

            train_loss_list = []
            train_acc_list = []
            test_acc_list = []

            # 准备训练数据
            keep = np.isin(train_label, keep_list[i])
            data = np.array(train_data)[keep]
            labels = np.array(train_label)[keep]

            # 准备测试数据
            test_keep = np.isin(test_label, keep_list[i])
            test_data_subset = np.array(test_data)[test_keep]
            test_labels_subset = np.array(test_label)[test_keep]

            test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(test_data_subset),
                torch.tensor(test_labels_subset)
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            weights.append(len(data) / all_len)

            gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
            gmm.fit(data)
            gmm_list.append(gmm)
            train_data_set = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels))
            train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                                            batch_size=BATCH_SIZE,
                                                            shuffle=True)

            best_test_acc = 0
            for epoch in range(EPOCH):
                # 训练阶段
                model.train()
                progress_bar = tqdm(train_data_loader, desc=f"Node {i} | Epoch {epoch}")
                epoch_loss = 0
                for x, y in progress_bar:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    output = model(x)
                    loss = loss_func(output, y)
                    train_loss_list.append(loss.item())
                    acc = acc_cal(output, y)
                    train_acc_list.append(acc)
                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    avg_loss = sum(train_loss_list[-len(train_data_loader):]) / len(train_data_loader)
                    avg_acc = sum(train_acc_list[-len(train_data_loader):]) / len(train_data_loader)
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

                # 测试阶段
                model.eval()
                test_acc = 0
                test_samples = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        output = model(x)
                        acc = acc_cal(output, y)
                        test_acc += acc * len(y)
                        test_samples += len(y)

                test_acc = test_acc / test_samples
                test_acc_list.append(test_acc)
                print(f"\nNode {i} | Epoch {epoch}")
                print(f"├─ Train Accuracy: {train_acc_list[-1]:.4f}")
                print(f"├─ Test Accuracy:  {test_acc:.4f}")
                print(f"└─ Best Test Accuracy so far: {best_test_acc:.4f}")

                # 更新最佳测试准确率和早停检查
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    # 保存最佳模型
                    torch.save(model.state_dict(), 
                                f'../result/model/mni_qffl_dp/best_noise_{noise}_clip_{l2_norm}_{NAME}_n{i}.pth')

                # 使用测试准确率进行早停检查
                if early_stopping(test_acc):
                    print(f"\nEarly stopping triggered:")
                    print(f"├─ Current epoch: {epoch}")
                    print(f"├─ Best test accuracy: {best_test_acc:.4f}")
                    print(f"└─ No improvement for {PATIENCE} epochs")
                    break

            # 记录最终结果
            node_results['accuracies'].append(train_acc_list[-1])
            node_results['test_accuracies'].append(best_test_acc)

            epsilon = moments_accountant(
                len(train_label),
                len(data),
                noise,
                EPOCH,
                1e-5
            )
            node_results['epsilons'].append(epsilon)

            print(f"\nNode {i} Final Results:")
            print(f"├─ Train Accuracy: {train_acc_list[-1]:.4f}")
            print(f"├─ Best Test Accuracy: {best_test_acc:.4f}")
            print(f"└─ Privacy Budget ε: {epsilon:.4f}")

            # 保存训练历史
            torch.save(train_loss_list,
                       f'../result/data/mni_qffl_dp/noise_{noise}_clip_{l2_norm}_{NAME}_train_loss_n{i}')
            torch.save(train_acc_list,
                       f'../result/data/mni_qffl_dp/noise_{noise}_clip_{l2_norm}_{NAME}_train_acc_n{i}')
            torch.save(test_acc_list,
                       f'../result/data/mni_qffl_dp/noise_{noise}_clip_{l2_norm}_{NAME}_test_acc_n{i}')

        # 计算并保存平均结果
        node_results['final_acc'] = np.mean(node_results['accuracies'])
        node_results['final_test_acc'] = np.mean(node_results['test_accuracies'])
        node_results['avg_epsilon'] = np.mean(node_results['epsilons'])

        print(f'\nConfiguration Summary - Noise: {noise}, Clip: {l2_norm}')
        print(f"├─ Average Train Accuracy: {node_results['final_acc']:.4f}")
        print(f"├─ Average Test Accuracy:  {node_results['final_test_acc']:.4f}"    )
        print(f"└─ Average Privacy Budget ε: {node_results['avg_epsilon']:.4f}\n")

        torch.save(gmm_list, f'../result/data/mni_qffl_dp/noise_{noise}_clip_{l2_norm}_{NAME}_gmm_list')
        torch.save(weights, f'../result/data/mni_qffl_dp/noise_{noise}_clip_{l2_norm}_{NAME}_data_weights')
        torch.save(node_results, f'../result/data/mni_qffl_dp/noise_{noise}_clip_{l2_norm}_{NAME}_results')

# 保存所有结果到JSON文件
results_dir = '../result/data/mni_qffl_dp'
os.makedirs(results_dir, exist_ok=True)
with open(f'{results_dir}/{NAME}_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f'Results saved to {results_dir}/{NAME}_results.json')
