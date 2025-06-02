import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from pyvacy import optim as pyvacy_optim
from pyvacy.analysis import moments_accountant

from mni_QFNN import Qfnn
from common.utils import acc_cal, setup_seed

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=300):
        self.buffer = []
        self.capacity = capacity

    def add(self, x, y):
        self.buffer.append((x.cpu().detach(), y.cpu().detach()))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# === 参数设置 ===
BATCH_SIZE = 128
EPOCH = 20
LR = 0.05
DEVICE = torch.device("cpu")
setup_seed(777)

noise_multiplier = 0.25
l2_norm_clip = 5.0
delta = 1e-5

NAME = "mnist_QFNN_FL_RDPCL_DP"
node = 5
keep_list = [[0,1],[2,3],[4,5],[6,7],[8,9]]

train_data = torch.load('../data/mnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/mnist/train_label.pkl').cpu().numpy()
all_len = len(train_label)

gmm_list = []
weights = []

for i in range(node):
    model = Qfnn(DEVICE).to(DEVICE)
    optimizer = pyvacy_optim.DPSGD(
        params=model.parameters(),
        lr=LR,
        batch_size=BATCH_SIZE,
        noise_multiplier=noise_multiplier,
        l2_norm_clip=l2_norm_clip
    )
    loss_func = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(capacity=300)

    # 筛选当前任务数据
    keep = np.isin(train_label, keep_list[i])
    data = np.array(train_data)[keep]
    labels = np.array(train_label)[keep]
    weights.append(len(data) / all_len)

    gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
    gmm.fit(data)
    gmm_list.append(gmm)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_loss_list, train_acc_list = [], []

    for epoch in range(EPOCH):
        model.train()
        progress = tqdm(train_loader, desc=f"Node {i} | Epoch {epoch}")
        for x_new, y_new in progress:
            x_new, y_new = x_new.to(DEVICE), y_new.to(DEVICE)
            output = model(x_new)
            loss = loss_func(output, y_new)

            # 加上 replay 内容（旧任务回顾）
            replay_batch = replay_buffer.sample(batch_size=64)
            for x_old, y_old in replay_batch:
                x_old, y_old = x_old.to(DEVICE), y_old.to(DEVICE)
                output_old = model(x_old)
                loss += 0.1 * loss_func(output_old, y_old)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = acc_cal(output, y_new)
            train_loss_list.append(loss.item())
            train_acc_list.append(acc)

            avg_loss = sum(train_loss_list[-len(train_loader):]) / len(train_loader)
            avg_acc = sum(train_acc_list[-len(train_loader):]) / len(train_loader)
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

            replay_buffer.add(x_new, y_new)

    epsilon = moments_accountant.compute_dp_sgd_privacy(
        n=all_len,
        batch_size=BATCH_SIZE,
        noise_multiplier=noise_multiplier,
        epochs=EPOCH,
        delta=delta
    )
    print(f"Node {i} | ε = {epsilon:.4f}")

    # 保存模型与训练记录
    os.makedirs(f'../result/model/mni_qffl_rdpcl', exist_ok=True)
    os.makedirs(f'../result/data/mni_qffl_rdpcl', exist_ok=True)
    torch.save(model.state_dict(), f'../result/model/mni_qffl_rdpcl/{NAME}_n{i}.pth')
    torch.save(train_loss_list, f'../result/data/mni_qffl_rdpcl/{NAME}_train_loss_n{i}.pt')
    torch.save(train_acc_list, f'../result/data/mni_qffl_rdpcl/{NAME}_train_acc_n{i}.pt')

torch.save(gmm_list, f'../result/data/mni_qffl_rdpcl/{NAME}_gmm_list')
torch.save(weights, f'../result/data/mni_qffl_rdpcl/{NAME}_data_weights')