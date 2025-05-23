import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp

# Model hyperparameters
n_qubits = 3
n_fuzzy_mem = 2
defuzz_layer = 2
# Number of defuzz wires = number of fuzzy rule activations
defuzz_qubits = n_fuzzy_mem ** n_qubits

# ===== T-Norm Quantum Circuit & QNN =====
# Input-only parameters
x_tnorm = ParameterVector('x_tnorm', n_qubits)
# Build T-norm circuit
num_wires_tnorm = 2 * n_qubits - 1
qc_tnorm = QuantumCircuit(num_wires_tnorm)
# Angle embedding
for i in range(n_qubits):
    qc_tnorm.ry(x_tnorm[i], i)
# Toffoli chain
qc_tnorm.ccx(0, 1, n_qubits)
for i in range(n_qubits - 2):
    qc_tnorm.ccx(i + 2, n_qubits + i, n_qubits + i + 1)
# Observable: projector onto |1> at target qubit
wire_p1 = 2 * n_qubits - 2
I_label = 'I' * num_wires_tnorm
Z_label = 'I' * wire_p1 + 'Z' + 'I' * (num_wires_tnorm - wire_p1 - 1)
obs_p1 = SparsePauliOp.from_list([(I_label, 0.5), (Z_label, -0.5)])
# Estimator backend
estimator = Estimator()
# QNN wrapper
qnn_tnorm = EstimatorQNN(
    circuit=qc_tnorm,
    observables=obs_p1,
    input_params=list(x_tnorm),
    weight_params=[],
    estimator=estimator
)
torch_tnorm = TorchConnector(qnn_tnorm)

# ===== Defuzz Quantum Circuit & QNN =====
# Input and weight parameters for defuzzifier
x_defuzz = ParameterVector('x_defuzz', defuzz_qubits)
w_defuzz = ParameterVector('w_defuzz', defuzz_layer * 3 * defuzz_qubits)
qc_defuzz = QuantumCircuit(defuzz_qubits)
# Angle embedding for defuzz inputs
for i in range(defuzz_qubits):
    qc_defuzz.ry(x_defuzz[i], i)
# Build observables list placeholder
obs_list = []
# Entangling + parameterized rotations
for layer in range(defuzz_layer):
    # Ring entanglement
    for j in range(defuzz_qubits - 1):
        qc_defuzz.cx(j, j + 1)
    qc_defuzz.cx(defuzz_qubits - 1, 0)
    # Parameterized rotations
    for j in range(defuzz_qubits):
        base = layer * 3 * defuzz_qubits + 3 * j
        qc_defuzz.rx(w_defuzz[base], j)
        qc_defuzz.rz(w_defuzz[base + 1], j)
        qc_defuzz.rx(w_defuzz[base + 2], j)
# Define observables for each wire
for j in range(defuzz_qubits):
    label = ''.join('Z' if k == j else 'I' for k in range(defuzz_qubits))
    obs_list.append(SparsePauliOp.from_list([(label, 1.0)]))
# QNN wrapper
eqnn_defuzz = EstimatorQNN(
    circuit=qc_defuzz,
    observables=obs_list,
    input_params=list(x_defuzz),
    weight_params=list(w_defuzz),
    estimator=estimator
)
torch_defuzz = TorchConnector(eqnn_defuzz)


# ===== Weight Initialization =====
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Parameter):
        torch.nn.init.normal_(m.data)


# ===== QFNN Model =====
class Qfnn(nn.Module):
    def __init__(self, device) -> None:
        super(Qfnn, self).__init__()
        self.device = device
        # Classical feature encoder
        self.linear = nn.Linear(10, n_qubits)
        self.dropout = nn.Dropout(0.5)
        # Fuzzy membership parameters
        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        # Defuzzifier output to 10 classes
        self.softmax_linear = nn.Linear(defuzz_qubits, 10)
        # Normalization layers
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(defuzz_qubits)
        self.apply(weights_init)
        # Quantum layers
        self.tnorm_layer = torch_tnorm
        self.defuzz_layer = torch_defuzz

    def forward(self, x):
        # Classical feature extraction
        x = self.linear(x)
        x = self.gn(x)
        # Fuzzification
        fuzzy0 = torch.zeros_like(x)
        fuzzy1 = torch.zeros_like(x)
        for i in range(x.shape[1]):
            a = -(x[:, i] - self.m[i, 0]) ** 2 / (2 * self.theta[i, 0] ** 2)
            b = -(x[:, i] - self.m[i, 1]) ** 2 / (2 * self.theta[i, 1] ** 2)
            fuzzy0[:, i] = torch.exp(a)
            fuzzy1[:, i] = torch.exp(b)
        fuzzy = torch.stack([fuzzy0, fuzzy1], dim=1)

        # T-norm quantum inference
        q_out = []
        for idx in range(n_fuzzy_mem ** n_qubits):
            loc = list(bin(idx)[2:])
            loc = [0] * (n_qubits - len(loc)) + loc
            q_in = torch.zeros_like(x)
            for j in range(n_qubits):
                q_in[:, j] = fuzzy[:, int(loc[j]), j]
            sq = torch.sqrt(q_in + 1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999)
            q_in = 2 * torch.arcsin(sq)
            p1 = self.tnorm_layer(q_in).squeeze(-1)
            q_out.append(p1)
        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)

        # Defuzzification quantum inference
        defuzz_out = self.defuzz_layer(out)
        # Final classification
        out = self.softmax_linear(defuzz_out)
        return out
