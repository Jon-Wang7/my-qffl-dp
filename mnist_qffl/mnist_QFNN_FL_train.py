import torch
import torch.nn as nn
from mni_QFNN import Qfnn
import numpy as np
from common.utils import acc_cal, setup_seed
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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
    # 打包数据和标签
    train_data_set = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels))
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True)

    for epoch in range(EPOCH):
        progress_bar = tqdm(train_data_loader, desc=f"Node {i} | Epoch {epoch}")
        for x, y in progress_bar:
            model.train()
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output = model(x)
            loss = loss_func(output, y)
            train_loss_list.append(loss.item())
            acc = acc_cal(output, y)
            train_acc_list.append(acc)
            avg_loss = sum(train_loss_list[-len(train_data_loader):]) / len(train_data_loader)
            avg_acc = sum(train_acc_list[-len(train_data_loader):]) / len(train_data_loader)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f'../result/model/mni_qffl/{NAME}_n{i}.pth')
    torch.save(train_loss_list, f'../result/data/mni_qffl/{NAME}_train_loss_n{i}')
    torch.save(train_acc_list, f'../result/data/mni_qffl/{NAME}_train_acc_n{i}')

torch.save(gmm_list, f'../result/data/mni_qffl/{NAME}_gmm_list')
torch.save(weights, f'../result/data/mni_qffl/{NAME}_data_weights')