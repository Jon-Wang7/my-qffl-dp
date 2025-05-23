import json
from qiskit_ibm_runtime import QiskitRuntimeService

# 初始化服务
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token="nnH4dAkS6t9MZC6NGKtVGiOjGL6nO_vONBAnOab5pTTo",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/60fc14240f4f4d83ba92b3c1b3ed6364:179a57bd-0ec3-4005-9a83-fdef8b7bd87d::"
)

job_id = "d0g7663gaiec73dr0uh0"
input_json = "qtnorm_theta_100x1.json"
output_json = "qtnorm_ibm_outputs_100x1.json"

# 加载 meta 信息
with open(input_json, "r") as f:
    meta = [json.loads(line.strip()) for line in f]

# 获取结果
job = service.job(job_id)
results = job.result()

# 保存为查表结构
with open(output_json, "w") as f:
    for i, pub_result in enumerate(results):
        counts = pub_result.join_data().get_counts()
        record = {
            "sample_idx": meta[i]["sample_idx"],
            "combination": meta[i]["combination"],
            "counts": counts
        }
        json.dump(record, f)
        f.write("\n")

print(f"✅ 已保存 IBM 真机测量结果，共 {len(results)} 条 → {output_json}")