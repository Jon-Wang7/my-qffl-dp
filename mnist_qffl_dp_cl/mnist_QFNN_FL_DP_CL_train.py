import os
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

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=500):
        self.buffer = []
        self.capacity = capacity

    def add(self, data, labels):
        noisy_data = (data.cpu() + torch.randn_like(data.cpu()) * 0.1).detach()
        labels = labels.cpu().detach()
        self.buffer.append((noisy_data, labels))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        actual_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), actual_size, replace=False)
        return [self.buffer[i] for i in indices]

# === 参数设置 ===
BATCH_SIZE = 128
EPOCH = 10
LR = 0.05
DEVICE = torch.device('cpu')
PATIENCE = 5
MIN_DELTA = 0.0005
MIN_EPOCHS = 3
setup_seed(777)

# === 数据加载 ===
train_data = torch.load('../data/mnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/mnist/train_label.pkl').cpu().numpy()
test_data = torch.load('../data/mnist/test_data.pkl').cpu().numpy()
test_label = torch.load('../data/mnist/test_label.pkl').cpu().numpy()

all_len = len(train_label)
gmm_list = []
weights = []

NAME = f'mnist_QFNN_gas_q4_star'
node = 5
keep_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

noise_multipliers = [0.1]
l2_norm_clips = [5]

results = {
    'noise_multipliers': noise_multipliers,
    'l2_norm_clips': l2_norm_clips,
    'node_results': {}
}

# === EarlyStopping 类 ===
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

# === ReplayBuffer 初始化 ===
replay_buffer = ReplayBuffer()

# === 联邦训练主循环 ===
for noise in noise_multipliers:
    for l2_norm in l2_norm_clips:
        node_results = {
            'accuracies': [],
            'epsilons': [],
            'final_acc': 0,
            'avg_epsilon': 0,
            'test_accuracies': [],
            'final_test_acc': 0
        }

        print(f'params: noise={noise}, l2_clip={l2_norm}')
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
            early_stopping = EarlyStopping()

            train_loss_list = []
            train_acc_list = []
            test_acc_list = []

            # 当前任务数据
            keep = np.isin(train_label, keep_list[i])
            data = np.array(train_data)[keep]
            labels = np.array(train_label)[keep]

            test_keep = np.isin(test_label, keep_list[i])
            test_data_subset = np.array(test_data)[test_keep]
            test_labels_subset = np.array(test_label)[test_keep]

            weights.append(len(data) / all_len)

            gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
            gmm.fit(data)
            gmm_list.append(gmm)

            train_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels)),
                batch_size=BATCH_SIZE, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(torch.tensor(test_data_subset), torch.tensor(test_labels_subset)),
                batch_size=BATCH_SIZE, shuffle=False
            )

            best_test_acc = 0
            for epoch in range(EPOCH):
                model.train()
                correct = 0
                total = 0
                progress_bar = tqdm(train_loader, desc=f"Node {i} | Epoch {epoch}")
                for x, y in progress_bar:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    output = model(x)
                    loss = loss_func(output, y)

                    # 加入 Replay 样本
                    if len(replay_buffer.buffer) > 0:
                        replay_samples = replay_buffer.sample(batch_size=32)
                        for x_old, y_old in replay_samples:
                            x_old, y_old = x_old.to(DEVICE), y_old.to(DEVICE)
                            loss += 0.3 * loss_func(model(x_old), y_old)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                    acc = correct / total
                    train_loss_list.append(loss.item())
                    train_acc_list.append(acc)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

                # 最后一批加入 Buffer
                replay_buffer.add(x, y)

                # 测试评估
                model.eval()
                test_acc = 0
                test_samples = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        output = model(x)
                        pred = output.argmax(dim=1)
                        test_acc += (pred == y).sum().item()
                        test_samples += y.size(0)
                test_acc = test_acc / test_samples
                test_acc_list.append(test_acc)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                if early_stopping(test_acc):
                    break

            # 记录结果
            node_results['accuracies'].append(train_acc_list[-1])
            node_results['test_accuracies'].append(best_test_acc)
            epsilon = moments_accountant(len(train_label), len(data), noise, EPOCH, 1e-5)
            node_results['epsilons'].append(epsilon)

            torch.save(train_loss_list, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{l2_norm}_{NAME}_train_loss_n{i}.pt')
            torch.save(train_acc_list, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{l2_norm}_{NAME}_train_acc_n{i}.pt')
            torch.save(test_acc_list, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{l2_norm}_{NAME}_test_acc_n{i}.pt')
            torch.save(model.state_dict(), f'../result/model/mni_qffl_dp_cl/last_noise_{noise}_clip_{l2_norm}_{NAME}_n{i}.pth')

        node_results['final_acc'] = np.mean(node_results['accuracies'])
        node_results['final_test_acc'] = np.mean(node_results['test_accuracies'])
        node_results['avg_epsilon'] = np.mean(node_results['epsilons'])

        print(f"\nSummary | noise={noise}, clip={l2_norm}")
        print(f"├─ Avg Train Acc: {node_results['final_acc']:.4f}")
        print(f"├─ Avg Test Acc:  {node_results['final_test_acc']:.4f}")
        print(f"└─ Avg ε:         {node_results['avg_epsilon']:.4f}")

        torch.save(gmm_list, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{l2_norm}_{NAME}_gmm_list')
        torch.save(weights, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{l2_norm}_{NAME}_data_weights')
        torch.save(node_results, f'../result/data/mni_qffl_dp_cl/noise_{noise}_clip_{l2_norm}_{NAME}_results')

# 保存 summary
results_dir = '../result/data/mni_qffl_dp_cl'
os.makedirs(results_dir, exist_ok=True)
with open(f'{results_dir}/{NAME}_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f'Results saved to {results_dir}/{NAME}_results.json')