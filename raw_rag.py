import os
# 解决网络连不上 HuggingFace 的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================================
# 第一步：准备资料（模拟公司的 3 个文档段落）
# ==========================================
print("1. 准备知识库...")
knowledge_base = [
    "公司年假规定：入职满一年有 5 天年假。",
    "差旅报销规定：高铁票需在月底前报销。",
    "考勤规定：迟到 3 次扣除全勤奖。"
]

# ==========================================
# 第二步：把文字变成坐标（Embedding）
# ==========================================
print("2. 正在把文字变成向量坐标，请稍候...")
# 加载昨天用的那个模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 把 3 句话全部变成 3 个坐标
# 结果 doc_vectors 是一个 3行512列 的矩阵
doc_vectors = model.encode(knowledge_base)

# ==========================================
# 第三步：检索（找小抄）
# ==========================================
# 员工提问
query = "如果我迟到了3次会怎么样？"
print(f"\n员工提问：{query}")

# 把问题也变成坐标
query_vector = model.encode(query)

# 计算余弦相似度（用 NumPy 手写，你学过的！）
# 1. 算点积
dot_products = np.dot(doc_vectors, query_vector)
# 2. 算长度
norm_docs = np.linalg.norm(doc_vectors, axis=1)
norm_query = np.linalg.norm(query_vector)
# 3. 算相似度
similarities = dot_products / (norm_docs * norm_query)

# 找到得分最高的那一段的索引（np.argmax 就是找最大值的索引）
best_index = np.argmax(similarities)
best_doc = knowledge_base[best_index]

print(f"--> 检索到最相关的段落是：{best_doc}")

# ==========================================
# 第四步：生成（老师看小抄答题）
# ==========================================
print("\n3. 正在呼叫大模型生成回答...")

# 把你申请的 API Key 填在这里
client = OpenAI(api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", base_url="https://api.deepseek.com")

# 核心：把检索到的资料（小抄）和问题拼在一起
prompt = f"""你是一个企业助手。
请严格根据以下资料回答员工问题。
资料：{best_doc}
问题：{query}
"""

# 调用大模型
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- AI 最终回答 ---")
print(response.choices[0].message.content)