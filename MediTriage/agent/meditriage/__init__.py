"""MediTriage 医疗 Agent 库。

子包：core（运行时内核）、swarm（多 Agent 编排）、agents（Worker 角色）、
knowledge（RAG）、memory（记忆分层）、research（深度研究）、guardrails（护栏）。
经 vLLM(:8000) + Milvus(:19530) 解耦；资产路径见 meditriage.paths。
"""
