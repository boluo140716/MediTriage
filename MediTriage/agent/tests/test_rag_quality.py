"""RAG 检索质量验证：中英文 + 跨语言 + rerank。"""



from meditriage.knowledge.milvus_kb import MedicalKnowledgeBase


def test_rag_quality():
    """冒烟测试：知识库检索全链路无异常即通过（需 Milvus 已启动）。"""
    kb = MedicalKnowledgeBase()
    print("total chunks:", kb.count_documents())

    queries = [
        ("高血压的一线降压药物", "中文→应命中 WHO高血压/中文指南"),
        ("atrial fibrillation anticoagulation therapy", "英文→应命中 ESC afib"),
        ("房颤的抗凝治疗", "跨语言：中文问→应命中英文 ESC afib"),
        ("COPD exacerbation management", "英文→应命中 GOLD copd"),
        ("慢性肾病的管理", "跨语言：中文问→应命中英文 KDIGO ckd"),
    ]
    for q, expect in queries:
        print(f"\n=== {q}  ({expect}) ===")
        results = kb.search(q, top_k=3)
        for r in results:
            m = r["metadata"]
            print(
                f"  [{r['score']:.3f}] "
                f"{m.get('source', '?')}/{m.get('topic', '?')}"
                f"({m.get('lang', '?')}): {r['content'][:90].strip()}"
            )


if __name__ == "__main__":
    test_rag_quality()
