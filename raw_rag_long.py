import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. 纯手写读取长文本并切块（按段落切分）
print("1. 正在读取长文本并切块...")
with open("long_rules.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 按照换行符切分，去掉空行
paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
print(f"文本被切分成了 {len(paragraphs)} 个段落。")

# 2. 把所有段落变成向量
print("2. 正在将段落转为向量...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
doc_vectors = model.encode(paragraphs) # 形状是 [段落数, 512]

# 3. 核心：Top-K 检索函数
def retrieve_top_k(query, paragraphs, doc_vectors, k=3):
    query_vector = model.encode(query)
    
    # 批量算余弦相似度
    dot_products = np.dot(doc_vectors, query_vector)
    norm_docs = np.linalg.norm(doc_vectors, axis=1)
    norm_query = np.linalg.norm(query_vector)
    similarities = dot_products / (norm_docs * norm_query)
    
    # 找到相似度最高的 K 个索引（np.argsort 会从小到大排，我们取最后 K 个，并反转）
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # 把找到的段落和得分拼起来返回
    results = []
    for idx in top_k_indices:
        results.append(paragraphs[idx])
    return results

# 4. 提问与生成
query = "如果我经常迟到会有什么惩罚？另外住宿费能报销多少？"
print(f"\n员工提问：{query}")

# 获取最相关的 3 个段落
top_k_docs = retrieve_top_k(query, paragraphs, doc_vectors, k=3)

print("\n---> 检索到的小抄如下：")
for i, doc in enumerate(top_k_docs):
    print(f"[{i+1}] {doc}")

# 把多段小抄拼成一段长文本
context = "\n".join(top_k_docs)

# 调用大模型
client = OpenAI(api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", base_url="https://api.deepseek.com")
prompt = f"""请严格根据以下资料回答问题。
资料：
{context}
问题：{query}
"""

print("\n3. 正在呼叫大模型生成回答...")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- AI 最终回答 ---")
print(response.choices[0].message.content)