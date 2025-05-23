
import json
import time
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.providers.jobstatus import JobStatus

api_token = "nnH4dAkS6t9MZC6NGKtVGiOjGL6nO_vONBAnOab5pTTo"
instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/60fc14240f4f4d83ba92b3c1b3ed6364:179a57bd-0ec3-4005-9a83-fdef8b7bd87d::"


# 初始化 IBM 后端服务
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=api_token,
    instance=instance_crn
)

# 选择后端
backend_name = "ibm_sherbrooke"  # 或者选择其他可用的后端
backend = service.backend(backend_name)

# 构建量子电路
def build_qtnorm_circuit(theta):
    n_qubits = 3
    total_qubits = 2 * n_qubits - 1  # =5
    qc = QuantumCircuit(total_qubits, total_qubits)

    for i in range(n_qubits):
        qc.ry(theta[i], i)

    qc.ccx(0, 1, n_qubits)
    for i in range(n_qubits - 2):
        qc.ccx(i + 2, n_qubits + i, i + n_qubits + 1)

    qc.measure(range(total_qubits), range(total_qubits))
    return qc

# 加载输入数据
input_json = "qtnorm_theta_100x1.json"
output_json = "qtnorm_ibm_outputs_100x1.json"
shots = 1024

records = []
with open(input_json, "r") as f:
    for line in f:
        records.append(json.loads(line))

# 构建所有电路
circuits = []
meta = []

print("📦 构建并转换电路中...")
for rec in records:
    circ = build_qtnorm_circuit(rec["theta"])
    transpiled_circ = transpile(circ, backend=backend)
    circuits.append(transpiled_circ)
    meta.append({"sample_idx": rec["sample_idx"], "combination": rec["combination"]})

# 初始化 Sampler
sampler = Sampler(mode=backend)

# 提交电路到 IBM 后端
print(f"⚛️ 提交 {len(circuits)} 个电路到 IBM 真机 {backend_name}...")
job = sampler.run(circuits)
print("🆔 Job ID:", job.job_id())
print("⌛ 正在等待量子任务执行...")
status = job.status()
while status not in ["DONE", "ERROR", "CANCELLED"]:
    print(f"当前状态: {status}")
    time.sleep(5)
    status = job.status()
print(f"✅ 最终状态: {status}")

# 读取结果
results = job.result()
counts = [res.data.meas.get_counts() for res in results]

# 保存结果为 JSON 格式
with open(output_json, "w") as f:
    for i, c in enumerate(counts):
        sample = {
            "sample_idx": meta[i]["sample_idx"],
            "combination": meta[i]["combination"],
            "counts": c
        }
        json.dump(sample, f)
        f.write("\n")

print(f"✅ 已保存 IBM 真机测量结果，共 {len(counts)} 条 → {output_json}")