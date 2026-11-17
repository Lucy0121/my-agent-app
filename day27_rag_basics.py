import numpy as np

# ==========================================
# 1. 准备知识库（假装这是公司制度文档）
# ==========================================
# 为了演示，我们假设每句话已经变成了 2 维的向量（真实情况是 1536 维）
# 这里我们简化处理，直接手写向量
documents = [
    {"text": "公司年假规定：入职满一年有 5 天年假。", "vector": np.array([0.8, 0.2])},
    {"text": "差旅报销规定：高铁票需在月底前报销。", "vector": np.array([0.1, 0.9])},
    {"text": "考勤规定：迟到 3 次扣除全勤奖。", "vector": np.array([0.5, 0.5])}
]

# ==========================================
# 2. 核心函数：余弦相似度（你昨天刚学的！）
# ==========================================
def cosine_similarity(query, doc):
    dot_product = np.dot(query, doc)
    norm_q = np.linalg.norm(query)
    norm_doc = np.linalg.norm(doc)
    return dot_product / (norm_q * norm_doc)

# ==========================================
# 3. 模拟员工提问
# ==========================================
# 员工问：“怎么报销火车票？”
# 假设这段问题也被转换成了向量
# query_vector = np.array([0.2, 0.8]) 
# print(f"员工提问：怎么报销火车票？\n对应向量：{query_vector}\n")

query_vector = np.array([0.8, 0.2]) 
print(f"员工提问：年假怎么休？\n对应向量：{query_vector}\n")

# ==========================================
# 4. 检索阶段（Retrieval）
# ==========================================
results = []
for doc in documents:
    # 算相似度
    score = cosine_similarity(query_vector, doc["vector"])
    results.append({"text": doc["text"], "score": score})

# 按得分从高到低排序
results.sort(key=lambda x: x["score"], reverse=True)

print("--- 检索结果排名 ---")
for res in results:
    print(f"得分: {res['score']:.4f} | 内容: {res['text']}")

# 取出第一名
best_match = results[0]["text"]

# ==========================================
# 5. 生成阶段（Generation）
# ==========================================
# 在真实系统中，这里会调用 ChatGPT 或 本地大模型 API
# 我们把查到的文档和问题拼在一起，假装大模型正在回答
print("\n--- 大模型最终回答 ---")
print(f"根据查到的资料【{best_match}】，您的火车票可以在月底前进行报销。")