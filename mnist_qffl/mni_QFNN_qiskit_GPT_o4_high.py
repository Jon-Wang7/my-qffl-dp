# 文件名: mni_QFNN_qiskit_final.py
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

# 预编译用的 statevector 模拟后端
backend_sim = AerSimulator(method="statevector", device="CPU")

# 定义常量
n_qubits = 3
n_fuzzy_mem = 2
defuzz_qubits = n_qubits  # 保持为3
defuzz_layer = 2


class Qfnn(nn.Module):
    def __init__(self, device) -> None:
        super(Qfnn, self).__init__()
        self.device = device
        self.linear = nn.Linear(10, n_qubits)
        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.softmax_linear = nn.Linear(defuzz_qubits, 10)
        # compression layer: move out of forward for efficiency
        self.compression = nn.Linear(n_fuzzy_mem ** n_qubits, defuzz_qubits)
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)
        self.apply(self._weights_init)

        self.quantum_tnorm = self._create_tnorm_qnn()
        self.quantum_defuzz = self._create_defuzz_qnn()

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m.data)

    def _create_tnorm_qnn(self):
        # 输入参数：3个角度参数
        input_params = ParameterVector("theta", length=n_qubits)
        # weight_params = ParameterVector("weights", length=weight_shapes["weights"][0])
        qc = QuantumCircuit(2 * n_qubits - 1)  # 5 qubits
        # 预编译量子电路以重用，提升前向/反向速度
        qc = transpile(qc, backend_sim)

        # 绑定输入参数到RY门
        for i in range(n_qubits):
            qc.ry(input_params[i], i)

        # Toffoli门结构（不引入新参数）
        qc.ccx(0, 1, n_qubits)
        for i in range(n_qubits - 2):
            qc.ccx(i + 2, n_qubits + i, i + n_qubits + 1)

        # 定义观测器（5个量子比特）
        num_qubits_circuit = qc.num_qubits
        observables = [
            SparsePauliOp(Pauli("I" * j + "Z" + "I" * (num_qubits_circuit - j - 1)))
            for j in range(num_qubits_circuit)
        ]

        # 关键修复：明确指定无权重参数
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=[],
            observables=observables,
            input_gradients=True,
            estimator=AerEstimator(  # 使用Aer的Estimator
                backend_options={
                    "method": "statevector",  # 或 "density_matrix", "matrix_product_state"
                    "device": "CPU"  # 启用GPU加速（需支持）
                },
                run_options={"shots": 1}  # 使用精确模拟
            ),
        )
        return TorchConnector(qnn)

    def _create_defuzz_qnn(self):
        # 输入参数：3个角度参数 + 权重参数：3*3*2=18
        input_params = ParameterVector("inputs", length=defuzz_qubits)
        weight_params = ParameterVector("weights", length=defuzz_layer * 3 * defuzz_qubits)
        qc = QuantumCircuit(defuzz_qubits)
        # 预编译量子电路以重用，提升前向/反向速度
        qc = transpile(qc, backend_sim)

        # 绑定输入参数到RY门
        for i in range(defuzz_qubits):
            qc.ry(input_params[i], i)

        # 绑定权重参数到量子门
        weight_idx = 0
        for i in range(defuzz_layer):
            for j in range(defuzz_qubits - 1):
                qc.cx(j, j + 1)
            qc.cx(defuzz_qubits - 1, 0)
            for j in range(defuzz_qubits):
                qc.rx(weight_params[weight_idx], j)
                qc.rz(weight_params[weight_idx + 1], j)
                qc.rx(weight_params[weight_idx + 2], j)
                weight_idx += 3

        # 定义观测器（3个量子比特）
        observables = [
            SparsePauliOp(Pauli("I" * j + "Z" + "I" * (defuzz_qubits - j - 1)))
            for j in range(defuzz_qubits)
        ]

        # 关键修复：正确绑定输入和权重参数
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,
            input_gradients=True,
            estimator=AerEstimator(  # 使用Aer的Estimator
                backend_options={
                    "method": "statevector",  # 或 "density_matrix", "matrix_product_state"
                    "device": "CPU"  # 启用GPU加速（需支持）
                },
                run_options={"shots": 1}  # 使用精确模拟
            ),
        )
        return TorchConnector(qnn)

    def forward(self, x):
        x = self.linear(x)
        x = self.gn(x)

        # 模糊化逻辑
        fuzzy_list0 = torch.zeros_like(x)
        fuzzy_list1 = torch.zeros_like(x)
        for i in range(x.shape[1]):
            a = (-(x[:, i] - self.m[i, 0]) ** 2) / (2 * self.theta[i, 0] ** 2)
            b = (-(x[:, i] - self.m[i, 1]) ** 2) / (2 * self.theta[i, 1] ** 2)
            fuzzy_list0[:, i] = torch.exp(a)
            fuzzy_list1[:, i] = torch.exp(b)
        fuzzy_list = torch.stack([fuzzy_list0, fuzzy_list1], dim=1)

        # 量子T-norm逻辑（输出8个特征）
        q_out = []
        for i in range(n_fuzzy_mem ** n_qubits):
            loc = format(i, f'0{n_qubits}b')
            q_in = torch.zeros_like(x)
            for j in range(n_qubits):
                q_in[:, j] = fuzzy_list[:, int(loc[j]), j]

            sq = torch.sqrt(q_in + 1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999)
            q_transformed = 2 * torch.arcsin(sq)
            Q_tnorm_out = self.quantum_tnorm(q_transformed)[:, 1]
            q_out.append(Q_tnorm_out)

        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)

        # 关键修复：通过预定义的 compression 层将8维压缩到3维
        out_compressed = self.compression(out)
        defuzz_out = self.quantum_defuzz(out_compressed)
        return self.softmax_linear(defuzz_out)