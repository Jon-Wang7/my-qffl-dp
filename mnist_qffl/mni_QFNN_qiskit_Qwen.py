import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler, Estimator
import math

# 定义常量
n_qubits = 3
n_fuzzy_mem = 2
defuzz_qubits = n_qubits
defuzz_layer = 2

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
        self.softmax_linear = nn.Linear(defuzz_qubits, 10)
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)
        self.apply(weights_init)

        # Initialize quantum layers
        self.sampler = Sampler()
        self.estimator = Estimator()

        # Define weight shapes for quantum layers
        self.weights1 = nn.Parameter(torch.randn(weight_shapes["weights"]))
        self.defuzz_weights = nn.Parameter(torch.randn(defuzz_weight_shapes["weights"]))

    def quantum_circuit1(self, inputs):
        batch_size = inputs.size(0)
        results = []

        for idx in range(batch_size):
            qc = QuantumCircuit(2 * n_qubits - 1, n_qubits)  # Add classical bits

            # Angle Embedding
            for i in range(n_qubits):
                angle = inputs[idx][i].item()  # Convert tensor to float
                qc.ry(angle, i)

            # Toffoli gates
            qc.ccx(0, 1, n_qubits)
            for i in range(n_qubits - 2):
                qc.ccx(i + 2, n_qubits + i, i + n_qubits + 1)

            # Measurement
            for i in range(n_qubits):
                qc.measure(i, i)

            # Execute the circuit using Sampler
            result = self.sampler.run(qc).result()
            quasis_probs = result.quasi_dists[0]
            probs = {int(k): v for k, v in quasis_probs.items()}
            results.append([probs.get(0, 0), probs.get(1, 0)])

        return torch.tensor(results, dtype=torch.float32, device=self.device)

    def quantum_circuit2(self, inputs):
        batch_size = inputs.size(0)
        results = []

        for idx in range(batch_size):
            qc = QuantumCircuit(defuzz_qubits, defuzz_qubits)  # Add classical bits

            # Amplitude Embedding
            norm = torch.norm(inputs[idx]).item()
            if norm != 0:
                inputs_normalized = inputs[idx] / norm
            else:
                inputs_normalized = inputs[idx]

            # Ensure the sum of squares is exactly 1
            norm_squared = torch.sum(inputs_normalized ** 2).item()
            if abs(norm_squared - 1.0) > 1e-6:
                inputs_normalized *= torch.sqrt(torch.tensor(1.0 / norm_squared)).to(inputs_normalized.device)

            # Force normalization by dividing by the L2 norm again
            inputs_normalized /= torch.norm(inputs_normalized)

            # Clip values to ensure numerical stability
            inputs_normalized = torch.clamp(inputs_normalized, min=-1.0, max=1.0)

            # Round to a fixed number of decimal places
            inputs_normalized = torch.round(inputs_normalized * 1e6) / 1e6

            # qc.initialize(inputs_normalized.tolist(), range(defuzz_qubits))

            fixed = inputs_normalized.detach().cpu().double()
            fixed = fixed / torch.norm(fixed, p=2)
            norm = torch.sum(fixed ** 2).item()

            if abs(norm - 1.0) > 1e-12:
                fixed = fixed / torch.sqrt(torch.tensor(norm))

            fixed = torch.round(fixed * 1e14) / 1e14  # 提高精度
            fixed = fixed / torch.norm(fixed, p=2)

            # 强制截断误差
            norm_final = torch.sum(fixed ** 2).item()
            fixed[-1] = fixed[-1] * (1.0 / math.sqrt(norm_final))  # 轻调最后一位

            qc.initialize(fixed.tolist(), range(defuzz_qubits))

            # Layers of CNOTs and RX/RZ gates
            for i in range(defuzz_layer):
                for j in range(defuzz_qubits - 1):
                    qc.cx(j, j + 1)
                qc.cx(defuzz_qubits - 1, 0)
                for j in range(defuzz_qubits):
                    rx_angle = self.defuzz_weights[i, 3 * j].item()  # Convert tensor to float
                    rz_angle = self.defuzz_weights[i, 3 * j + 1].item()  # Convert tensor to float
                    next_rx_angle = self.defuzz_weights[i, 3 * j + 2].item()  # Convert tensor to float
                    qc.rx(rx_angle, j)
                    qc.rz(rz_angle, j)
                    qc.rx(next_rx_angle, j)

            # Measurement
            for i in range(defuzz_qubits):
                qc.measure(i, i)

            # Execute the circuit using Sampler
            result = self.sampler.run(qc, shots=1024).result()
            counts = result.quasi_dists[0]

            # Calculate expectation values
            expvals = []
            for i in range(defuzz_qubits):
                expval_z = (counts.get('0' * i + '0' + '0' * (defuzz_qubits - i - 1), 0) -
                            counts.get('0' * i + '1' + '0' * (defuzz_qubits - i - 1), 0)) / 1024
                expvals.append(expval_z)

            results.append(expvals)

        return torch.tensor(results, dtype=torch.float32, device=self.device)

    def forward(self, x):
        device = self.device
        x = self.linear(x)
        x = self.gn(x)
        fuzzy_list0 = torch.zeros_like(x).to(device)
        fuzzy_list1 = torch.zeros_like(x).to(device)
        for i in range(x.shape[1]):
            a = (-(x[:, i] - self.m[i, 0]) ** 2) / (2 * self.theta[i, 0] ** 2)
            b = (-(x[:, i] - self.m[i, 1]) ** 2) / (2 * self.theta[i, 1] ** 2)
            fuzzy_list0[:, i] = torch.exp(a)
            fuzzy_list1[:, i] = torch.exp(b)

        fuzzy_list = torch.stack([fuzzy_list0, fuzzy_list1], dim=1)

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
            Q_tnorm_out = self.quantum_circuit1(q_in)[:, 1]
            q_out.append(Q_tnorm_out)

        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)

        defuzz_out = self.quantum_circuit2(out)
        out = self.softmax_linear(defuzz_out)
        return out
