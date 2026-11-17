import numpy as np

# 1. 假设我们有 3 句话（已经变成了向量）
# 句子0: "我喜欢苹果"（水果） -> [0.1, 0.9]
# 句子1: "我想买iPhone"（科技） -> [0.9, 0.1]
# 句子2: "苹果手机很好用"（科技偏多） -> [0.8, 0.2]
documents = np.array([
    [0.1, 0.9], 
    [0.9, 0.1],
    [0.8, 0.2]
])

# 用户搜索："苹果手机"
query = np.array([0.8, 0.2])

# 2. 核心公式：余弦相似度（用纯数学公式实现）
def cosine_similarity(query, doc):
    # 点积
    dot_product = np.dot(query, doc)
    # 长度
    norm_q = np.linalg.norm(query)
    norm_doc = np.linalg.norm(doc)
    # 套公式
    return dot_product / (norm_q * norm_doc)

# 3. 挨个算一遍
print("--- 检索结果 ---")
for i, doc in enumerate(documents):
    score = cosine_similarity(query, doc)
    print(f"句子 {i} 的余弦相似度: {score:.4f}")

# 4. 你会发现句子 2（苹果手机很好用）的分数最高，这就是 RAG 检索的最底层原理！