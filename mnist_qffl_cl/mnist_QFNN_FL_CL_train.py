import os
import torch
import torch.nn as nn
from mni_QFNN import Qfnn
import numpy as np
from common.utils import acc_cal, setup_seed
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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
        if actual_size == 0:
            return []
        indices = np.random.choice(len(self.buffer), actual_size, replace=False)
        return [self.buffer[i] for i in indices]

BATCH_SIZE = 128
EPOCH = 5
LR = 0.1
DEVICE = torch.device('cpu')

setup_seed(777)

transform = transforms.Compose([transforms.ToTensor()])
train_data = torch.load('../data/mnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/mnist/train_label.pkl').cpu().numpy()

all_len = len(train_label)
gmm_list = []
weights = []

NAME = f'mnist_QFNN_gas_q4_star'
node = 5
keep_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

replay_buffer = ReplayBuffer()

for i in range(node):
    model = Qfnn(DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []

    keep = np.isin(train_label, keep_list[i])
    data = np.array(train_data)[keep]
    labels = np.array(train_label)[keep]
    weights.append(len(data) / all_len)

    gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
    gmm.fit(data)
    gmm_list.append(gmm)

    train_data_set = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels))
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        progress_bar = tqdm(train_data_loader, desc=f"Node {i} | Epoch {epoch}")
        correct = 0
        total = 0

        for x, y in progress_bar:
            model.train()
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # === 合并当前 batch 和 replay 样本 ===
            if len(replay_buffer.buffer) > 0:
                replay_samples = replay_buffer.sample(batch_size=32)
                x_old, y_old = zip(*replay_samples)
                x_old = torch.cat(x_old).to(DEVICE)
                y_old = torch.cat(y_old).to(DEVICE)

                # 合并当前任务数据 + replay 数据
                x_combined = torch.cat([x, x_old], dim=0)
                y_combined = torch.cat([y, y_old], dim=0)
            else:
                x_combined = x
                y_combined = y

            output = model(x_combined)
            loss = loss_func(output, y_combined)

            pred = output.argmax(dim=1)
            correct += (pred == y_combined).sum().item()
            total += y_combined.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = correct / total
            train_loss_list.append(loss.item())
            train_acc_list.append(acc)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

            # === 每次 step 后都加当前样本进 buffer（而不是只最后）===
            replay_buffer.add(x, y)

    torch.save(model.state_dict(), f'../result/model/mni_qffl_cl/{NAME}_n{i}.pth')
    torch.save(train_loss_list, f'../result/data/mni_qffl_cl/{NAME}_train_loss_n{i}')
    torch.save(train_acc_list, f'../result/data/mni_qffl_cl/{NAME}_train_acc_n{i}')

torch.save(gmm_list, f'../result/data/mni_qffl_cl/{NAME}_gmm_list')
torch.save(weights, f'../result/data/mni_qffl_cl/{NAME}_data_weights')