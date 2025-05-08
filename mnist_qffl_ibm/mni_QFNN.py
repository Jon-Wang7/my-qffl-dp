import torch
import numpy as np
import pennylane as qml
from qiskit_ibm_provider import IBMProvider

# IBM 配置
API_TOKEN = "k50IjcqonwxKgRmDCEwWHJXClm9oUVH5OEbWxLpUhhpf"
provider = IBMProvider(token=API_TOKEN)
backend = provider.get_backend("ibmq_qasm_simulator")

# 设备与电路设定
n_qubits = 3
dev1 = qml.device('qiskit.ibmq', wires=2 * n_qubits - 1, backend='ibmq_qasm_simulator', shots=1000)

@qml.qnode(dev1, interface="torch")
def q_tnorm_node(inputs):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.Toffoli(wires=[0, 1, n_qubits])
    for i in range(n_qubits - 2):
        qml.Toffoli(wires=[i + 2, n_qubits + i, i + n_qubits + 1])
    return qml.probs(wires=2 * n_qubits - 2)

# 采样执行
samples = 100
results_tnorm = []

for _ in range(samples):
    x = torch.tensor(np.random.uniform(0, np.pi, size=(n_qubits,)), dtype=torch.float32)
    output = q_tnorm_node(x)
    results_tnorm.append(output.detach().numpy())

np.save("q_tnorm_ibmq.npy", np.array(results_tnorm))
print("✅ tnorm 电路输出已保存到 q_tnorm_ibmq.npy")