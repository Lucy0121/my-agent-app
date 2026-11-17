import os
# 强制设置 HuggingFace 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ==========================================
# 1. 准备阶段（你昨天写的检索逻辑）
# ==========================================
print("正在加载 Embedding 模型...")
embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

documents = [
    "公司年假规定：入职满一年有 5 天年假。",
    "差旅报销规定：高铁票需在月底前报销。",
    "考勤规定：迟到 3 次扣除全勤奖。"
]
doc_embeddings = embed_model.encode(documents)

# 员工提问
query = "我坐高铁出差的票怎么报销？"
query_embedding = embed_model.encode(query)

# 检索最相关的文档
scores = cos_sim(query_embedding, doc_embeddings)[0]
best_doc_index = np.argmax(scores) # 找到得分最高的索引
best_doc = documents[best_doc_index]

print(f"\n[检索阶段] 查到的相关制度：{best_doc}")

# ==========================================
# 2. 生成阶段（调用大模型 API）
# ==========================================
# 填入你刚才申请的 API Key（注意保留引号）
client = OpenAI(
    api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", 
    base_url="https://api.deepseek.com" # DeepSeek的接口地址
)

# 核心逻辑：把检索到的资料和问题拼成一段话（Prompt），交给大模型
prompt = f"""
你是一个专业的企业内部助手。
请严格根据以下提供的资料，用礼貌的语气回答员工的问题。不要自己编造资料以外的东西。

【资料】：
{best_doc}

【员工问题】：
{query}
"""

print("\n[生成阶段] 大模型正在思考...")
# 调用大模型
response = client.chat.completions.create(
    model="deepseek-chat", # 模型名字
    messages=[
        {"role": "system", "content": "你是一个严谨的AI助手。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3 # 温度调低，让它老老实实根据资料回答，不乱编
)

# 打印大模型的回答
print("\n--- AI 最终回答 ---")
print(response.choices[0].message.content)