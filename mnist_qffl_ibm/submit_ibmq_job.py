from qiskit_ibm_runtime import QiskitRuntimeService
import pennylane as qml
import torch
import numpy as np

# === 替换为你的真实 API Token ===
API_TOKEN = "k50IjcqonwxKgRmDCEwWHJXClm9oUVH5OEbWxLpUhhpf"  # ← 改成你实际的 token

# ✅ 使用新版 ibm_cloud 通道（官方推荐）
service = QiskitRuntimeService(channel="ibm_cloud", token=API_TOKEN)
backend = service.backend(name="ibmq_qasm_simulator")

# ========= 配置电路 ==========
n_qubits = 3
n_total_wires = 2 * n_qubits - 1
dev_tnorm = qml.device('qiskit.ibmq', wires=n_total_wires, backend='ibmq_qasm_simulator', shots=1000)

@qml.qnode(dev_tnorm, interface="torch")
def q_tnorm_node(inputs):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.Toffoli(wires=[0, 1, n_qubits])
    for i in range(n_qubits - 2):
        qml.Toffoli(wires=[i + 2, n_qubits + i, i + n_qubits + 1])
    return qml.probs(wires=2 * n_qubits - 2)

# ========= 采样并保存 ==========
samples = 100
results_tnorm = []

for _ in range(samples):
    x = torch.tensor(np.random.uniform(0, np.pi, size=(n_qubits,)), dtype=torch.float32)
    output = q_tnorm_node(x)
    results_tnorm.append(output.detach().numpy())

np.save("q_tnorm_ibmq.npy", np.array(results_tnorm))
print("✅ 已完成采样并保存至 q_tnorm_ibmq.npy")