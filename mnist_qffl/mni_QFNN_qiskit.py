import numpy as np

import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

n_qubits = 3

n_fuzzy_mem = 2
# device='cuda:0'
defuzz_qubits = n_qubits
defuzz_layer = 2


def q_tnorm_node(inputs):
    results = []
    for input_vec in inputs:
        input_vec = input_vec.detach().cpu().numpy()
        qc = QuantumCircuit(5)
        for i in range(3):
            qc.ry(input_vec[i], i)
        qc.ccx(0, 1, 3)
        qc.ccx(2, 3, 4)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        results.append(probs)
    return torch.tensor(np.array(results), dtype=torch.float32)

def q_defuzz(inputs):
    results = []
    for input_vec in inputs:
        input_vec = input_vec.detach().cpu().numpy()

        # 只取前 3 个作为输入（代表 low/med/high）
        vec3 = input_vec[:3]

        qc = QuantumCircuit(3)
        for i in range(3):
            qc.ry(vec3[i], i)  # 用 RY 角度编码每个 fuzzy 概念

        # 可选 entanglement（略）
        # qc.cx(0, 1)
        # qc.cx(1, 2)

        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()  # 长度 = 8

        # 只取前三个 basis 状态 |000>, |001>, |010> 作为 low/med/high 的概率
        # 对应状态索引：0 (000), 1 (001), 2 (010)
        defuzzed = [probs[0], probs[1], probs[2]]
        results.append(defuzzed)

    return torch.from_numpy(np.array(results)).float().to(inputs.device).requires_grad_()


weight_shapes = {"weights": (1, 1)}
defuzz_weight_shapes = {"weights": (defuzz_layer, 3 * defuzz_qubits)}


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Parameter):
        torch.nn.init.normal_(m.data)


class Qfnn(nn.Module):
    def __init__(self, device) -> None:
        super(Qfnn, self).__init__()
        self.device = device
        self.linear = nn.Linear(10, n_qubits)
        self.dropout = nn.Dropout(0.5)
        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        # self.linear2=nn.Linear(n_fuzzy_mem**n_qubits,10)
        self.softmax_linear = nn.Linear(n_fuzzy_mem ** n_qubits, 10)
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)
        self.apply(weights_init)

        # self.qlayer = qml.qnn.TorchLayer(q_tnorm_node, weight_shapes)
        self.defuzz = lambda x: q_defuzz(x)

    def forward(self, x):
        device = self.device
        x = self.linear(x)
        # x=nn.ReLU()(x)
        # x=self.dropout(x)
        # 规定为正数
        # min=torch.min(x)
        # max=torch.max(x)
        # x=(x-min)/(max-min)
        x = self.gn(x)
        fuzzy_list0 = torch.zeros_like(x).to(device)
        fuzzy_list1 = torch.zeros_like(x).to(device)
        for i in range(x.shape[1]):
            a = (-(x[:, i] - self.m[i, 0]) ** 2) / (2 * self.theta[i, 0] ** 2)
            b = (-(x[:, i] - self.m[i, 1]) ** 2) / (2 * self.theta[i, 1] ** 2)
            fuzzy_list0[:, i] = torch.exp(a)
            fuzzy_list1[:, i] = torch.exp(b)

        fuzzy_list = torch.stack([fuzzy_list0, fuzzy_list1], dim=1)

        # fuzzy_list=self.bn(fuzzy_list)
        q_in = torch.zeros_like(x).to(device)
        q_out = []
        for i in range(n_fuzzy_mem ** n_qubits):
            loc = list(bin(i))[2:]
            if len(loc) < n_qubits:
                loc = [0] * (n_qubits - len(loc)) + loc
            for j in range(n_qubits):
                q_in = q_in.clone()
                q_in[:, j] = fuzzy_list[:, int(loc[j]), j]

            sq = torch.sqrt(q_in + 1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999)
            q_in = 2 * torch.arcsin(sq)
            # q_in=q_in.clone()
            # Q_tnorm_out = self.qlayer(q_in)[:, 1]
            Q_tnorm_probs = q_tnorm_node(q_in)
            Q_tnorm_out = Q_tnorm_probs[:, 1]
            q_out.append(Q_tnorm_out)
            # 将q_in的每一列相乘
            # out_cheng=torch.prod(q_in,dim=1)
            # q_out.append(out_cheng)
        # q_out=nn.ReLU()(q_out)
        # q_out=self.dropout(q_out)
        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)

        # out=self.linear2(out)
        out = self.softmax_linear(out)
        out = self.defuzz(out)
        return out

# if __name__ == "__main__":
#     x=torch.randn(20,10)
#     model=Qfnn('cpu')
#     out=model(x)
#     print(out.shape)
