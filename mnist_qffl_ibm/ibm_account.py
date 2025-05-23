# from qiskit_ibm_runtime import QiskitRuntimeService
#
# token = "nnH4dAkS6t9MZC6NGKtVGiOjGL6nO_vONBAnOab5pTTo"  # 替换为您的实际 token
# QiskitRuntimeService.save_account(channel="ibm_quantum", token=token)



from qiskit_ibm_runtime import QiskitRuntimeService

# 使用您的 IBM Cloud API 密钥和实例 CRN 初始化服务
api_token = "nnH4dAkS6t9MZC6NGKtVGiOjGL6nO_vONBAnOab5pTTo"
instance_crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/60fc14240f4f4d83ba92b3c1b3ed6364:179a57bd-0ec3-4005-9a83-fdef8b7bd87d::"

service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=api_token,
    instance=instance_crn
)

# 列出可用的后端
backends = service.backends()
for backend in backends:
    print(backend.name)