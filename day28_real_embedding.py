from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 加载一个专门处理中文的轻量级 Embedding 模型（BGE 是国产非常牛的模型）
# 第一次运行会自动下载，大概几百兆
print("正在加载模型，请稍候...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 2. 真实的文本数据
documents = [
    "公司年假规定：入职满一年有 5 天年假。",
    "差旅报销规定：高铁票需在月底前报销。",
    "考勤规定：迟到 3 次扣除全勤奖。"
]

# 3. 核心步骤：把文档变成向量矩阵（形状: [3, 512]）
# 真实模型的维度通常是 512 或 768 或 1536，不再是 2 维了
doc_embeddings = model.encode(documents)
print(f"文档矩阵形状: {doc_embeddings.shape}")

# 4. 员工提问
query = "怎么报销火车票？"
query_embedding = model.encode(query)

# 5. 计算相似度（用 SentenceTransformer 自带的余弦相似度计算工具）
from sentence_transformers.util import cos_sim

# 计算 query 与 3 个文档的相似度
scores = cos_sim(query_embedding, doc_embeddings)[0]
print(f"\n问题：【{query}】")

# 6. 排序并输出
for i, score in enumerate(scores):
    print(f"相似度: {score.item():.4f} | 文档: {documents[i]}")