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


class LearnableTnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_qubits))
        self.shift = np.pi / 2

    def forward(self, inputs):
        results = []
        for input_vec in inputs:
            probs_list = []
            for i in range(n_qubits):
                shifted_pos = self.theta.detach().clone()
                shifted_neg = self.theta.detach().clone()
                shifted_pos[i] += self.shift
                shifted_neg[i] -= self.shift

                prob_pos = self._run_circuit(input_vec, shifted_pos)
                prob_neg = self._run_circuit(input_vec, shifted_neg)
                grad = (prob_pos - prob_neg) / 2
                probs_list.append((prob_pos + prob_neg) / 2)
            avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
            results.append(avg_probs)
        return torch.stack(results, dim=0).to(inputs.device).requires_grad_()

    def _run_circuit(self, input_vec, theta_vals):
        qc = QuantumCircuit(5)
        input_vec = input_vec.detach().cpu().numpy()
        for i in range(n_qubits):
            angle = theta_vals[i].item() * input_vec[i]
            qc.ry(angle, i)
        qc.ccx(0, 1, 3)
        qc.ccx(2, 3, 4)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        return torch.tensor(probs, dtype=torch.float32)


class LearnableDefuzz(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(3))
        self.shift = np.pi / 2

    def forward(self, input_vec):
        results = []
        for _ in input_vec:
            gradients = []
            outputs = []
            for i in range(3):
                shifted_theta_pos = self.theta.detach().clone()
                shifted_theta_neg = self.theta.detach().clone()
                shifted_theta_pos[i] += self.shift
                shifted_theta_neg[i] -= self.shift

                probs_pos = self._run_circuit(shifted_theta_pos)
                probs_neg = self._run_circuit(shifted_theta_neg)
                grad_i = (probs_pos - probs_neg) / 2
                gradients.append(grad_i)
                outputs.append((probs_pos + probs_neg) / 2)

            avg_out = torch.stack(outputs, dim=0).mean(dim=0)
            results.append(avg_out)

        return torch.stack(results, dim=0).to(input_vec.device).requires_grad_()

    def _run_circuit(self, theta_vals):
        qc = QuantumCircuit(3)
        for i in range(3):
            qc.ry(theta_vals[i].item(), i)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        return torch.tensor([probs[0], probs[1], probs[2]], dtype=torch.float32)


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
        self.defuzz = LearnableDefuzz()
        self.tnorm = LearnableTnorm()

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
            Q_tnorm_probs = self.tnorm(q_in)
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
