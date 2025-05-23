# 文件名: mni_QFNN_qiskit_DS_v2.py
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

# 定义常量
n_qubits = 3
n_fuzzy_mem = 2
defuzz_qubits = n_qubits  # 保持为3
defuzz_layer = 2
BATCH_SIZE = 32  # 添加批处理大小常量

# 定义观测器
T_NORM_OBSERVABLE = [SparsePauliOp(Pauli("Z" + "I" * (2 * n_qubits - 2)))]  
DEFUZZ_OBSERVABLES = [SparsePauliOp(Pauli("I" * j + "Z" + "I" * (defuzz_qubits - j - 1))) for j in range(defuzz_qubits)]

# 定义后端选项
BACKEND_OPTIONS = {
    "method": "statevector",
    "device": "CPU",
    "max_parallel_threads": 4,  # 最大并行线程数
    "max_parallel_experiments": 1,  # 最大并行实验数
    "max_parallel_shots": 1  # 最大并行shots数
}

class Qfnn(nn.Module):
    def __init__(self, device) -> None:
        super(Qfnn, self).__init__()
        self.device = device
        
        # 经典神经网络层
        self.linear = nn.Linear(10, n_qubits)
        self.softmax_linear = nn.Linear(defuzz_qubits, 10)
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)
        
        # 模糊化参数
        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        
        # 初始化权重
        self.apply(self._weights_init)
        
        # 量子层
        self.quantum_tnorm = self._create_tnorm_qnn()
        self.quantum_defuzz = self._create_defuzz_qnn()
        
        # 缓存量子电路
        self._setup_cache()

    def _setup_cache(self):
        """设置缓存以提高性能"""
        # 初始化缓存
        self.tnorm_cache = {}
        self.defuzz_cache = {}
        
        # 预分配内存
        self.batch_size = 32  # 默认批大小
        self.fuzzy_list0 = torch.zeros((self.batch_size, n_qubits), device=self.device)
        self.fuzzy_list1 = torch.zeros((self.batch_size, n_qubits), device=self.device)
        self.q_in = torch.zeros((self.batch_size, n_qubits), device=self.device)
        self.sqrt_eps = torch.tensor(1e-16, device=self.device)
        self.upper_bound = torch.tensor(0.99999, device=self.device)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m.data)

    def _create_tnorm_circuit(self):
        """创建简化的T-norm量子电路"""
        qc = QuantumCircuit(2 * n_qubits - 1)
        input_params = ParameterVector("theta", length=n_qubits)
        
        # 简化的角度编码
        for i in range(n_qubits):
            qc.ry(input_params[i], i)

        # 简化的Toffoli门结构
        qc.ccx(0, 1, n_qubits)
        for i in range(n_qubits - 2):
            qc.ccx(i + 2, n_qubits + i, i + n_qubits + 1)
            
        return qc

    def _create_defuzz_circuit(self):
        """创建简化的去模糊化量子电路"""
        qc = QuantumCircuit(defuzz_qubits)
        input_params = ParameterVector("inputs", length=defuzz_qubits)
        weight_params = ParameterVector("weights", length=defuzz_layer * 3 * defuzz_qubits)
        
        # 简化的角度编码
        for i in range(defuzz_qubits):
            qc.ry(input_params[i], i)

        # 简化的参数化量子层
        weight_idx = 0
        for i in range(defuzz_layer):
            # 简化的纠缠层
            for j in range(defuzz_qubits - 1):
                qc.cx(j, j + 1)
            qc.cx(defuzz_qubits - 1, 0)
            
            # 简化的参数化旋转层
            for j in range(defuzz_qubits):
                qc.rx(weight_params[weight_idx], j)
                qc.rz(weight_params[weight_idx + 1], j)
                qc.rx(weight_params[weight_idx + 2], j)
                weight_idx += 3
                
        return qc

    def _create_tnorm_qnn(self):
        """创建简化的量子T-norm神经网络"""
        qc = self._create_tnorm_circuit()
        
        # 使用单个观测器
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=qc.parameters[:n_qubits],
            weight_params=[],
            observables=T_NORM_OBSERVABLE,
            input_gradients=True,
            estimator=AerEstimator(
                backend_options=BACKEND_OPTIONS,
                run_options={"shots": 1}
            ),
        )
        return TorchConnector(qnn)

    def _create_defuzz_qnn(self):
        """创建简化的量子去模糊化神经网络"""
        qc = self._create_defuzz_circuit()
        
        # 使用单个观测器
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=qc.parameters[:defuzz_qubits],
            weight_params=qc.parameters[defuzz_qubits:],
            observables=DEFUZZ_OBSERVABLES,
            input_gradients=True,
            estimator=AerEstimator(
                backend_options=BACKEND_OPTIONS,
                run_options={"shots": 1}
            ),
        )
        return TorchConnector(qnn)

    def _fuzzy_process(self, x):
        """优化的模糊化处理"""
        batch_size = x.shape[0]
        
        # 如果批大小变化，重置内存
        if batch_size != self.fuzzy_list0.shape[0]:
            self.fuzzy_list0 = torch.zeros((batch_size, n_qubits), device=self.device)
            self.fuzzy_list1 = torch.zeros((batch_size, n_qubits), device=self.device)
            self.batch_size = batch_size
        else:
            self.fuzzy_list0.zero_()
            self.fuzzy_list1.zero_()
        
        # 向量化计算
        for i in range(x.shape[1]):
            diff0 = x[:, i] - self.m[i, 0]
            diff1 = x[:, i] - self.m[i, 1]
            self.fuzzy_list0[:, i] = torch.exp(-(diff0 ** 2) / (2 * self.theta[i, 0] ** 2))
            self.fuzzy_list1[:, i] = torch.exp(-(diff1 ** 2) / (2 * self.theta[i, 1] ** 2))
            
        return torch.stack([self.fuzzy_list0, self.fuzzy_list1], dim=1)

    def _quantum_tnorm_process(self, fuzzy_list):
        """优化的量子T-norm处理"""
        q_out = []
        batch_size = fuzzy_list.shape[0]
        
        # 如果批大小变化，重置内存
        if batch_size != self.q_in.shape[0]:
            self.q_in = torch.zeros((batch_size, n_qubits), device=self.device)
            self.batch_size = batch_size
        else:
            self.q_in.zero_()
        
        # 计算所有可能的组合
        for i in range(n_fuzzy_mem ** n_qubits):
            loc = format(i, f'0{n_qubits}b')
            for j in range(n_qubits):
                self.q_in[:, j] = fuzzy_list[:, int(loc[j]), j]

            # 优化的变换
            sq = torch.sqrt(torch.clamp(self.q_in + self.sqrt_eps, 0.0, self.upper_bound))
            q_transformed = 2 * torch.arcsin(sq)
            
            # 使用缓存来计算量子T-norm输出
            q_key = str(q_transformed.flatten().cpu().detach().numpy().round(4).tolist())
            if q_key in self.tnorm_cache and not self.training:
                Q_tnorm_out = self.tnorm_cache[q_key]
            else:
                Q_tnorm_out = self.quantum_tnorm(q_transformed)[:, 0]
                if not self.training:
                    self.tnorm_cache[q_key] = Q_tnorm_out
                    
            q_out.append(Q_tnorm_out)
            
        return q_out

    def forward(self, x):
        # 经典神经网络处理
        x = self.linear(x)
        x = self.gn(x)

        # 模糊化处理
        fuzzy_list = self._fuzzy_process(x)

        # 量子T-norm处理
        q_out = self._quantum_tnorm_process(fuzzy_list)
        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)

        # 量子去模糊化
        out_compressed = nn.Linear(8, 3, device=self.device)(out)
        
        # 使用缓存来计算去模糊化输出
        q_key = str(out_compressed.flatten().cpu().detach().numpy().round(4).tolist())
        if q_key in self.defuzz_cache and not self.training:
            defuzz_out = self.defuzz_cache[q_key]
        else:
            defuzz_out = self.quantum_defuzz(out_compressed)
            if not self.training:
                self.defuzz_cache[q_key] = defuzz_out
        
        # 确保defuzz_out的维度正确
        if len(defuzz_out.shape) == 1:
            defuzz_out = defuzz_out.unsqueeze(1)
        
        return self.softmax_linear(defuzz_out)