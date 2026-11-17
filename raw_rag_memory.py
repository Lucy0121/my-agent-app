import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 初始化
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
client = OpenAI(api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", base_url="https://api.deepseek.com")

# 1. 准备一个非常短的资料库
knowledge_base = [
    "考勤规定：迟到 3 次扣除全勤奖。",
    "请假规定：需提前一天在 OA 系统提交申请，病假需提供证明。",
    "年假规定：入职满一年有 5 天年假。",
    "报销规定：高铁票需在月底前报销。"
]
doc_vectors = model.encode(knowledge_base)

# 2. 准备一个“记忆盒子”
chat_history = []

# 3. 核心函数：重写问题（让它带上之前的上下文）
def rewrite_query(user_query, history):
    if not history:
        return user_query # 如果没历史，直接返回原问题
    
    # 把历史记录拼成字符串
    history_text = "\n".join([f"用户：{h['user']}\nAI：{h['ai']}" for h in history])
    
    prompt = f"""根据以下历史对话，将用户的最新提问改写成一个完整的、独立的、不需要上下文也能看懂的问题。
    不要回答问题，只输出改写后的那一句话。
    
    历史对话：
    {history_text}
    
    用户最新提问：{user_query}
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 4. 核心函数：检索
def retrieve(query, k=1):
    query_vector = model.encode(query)
    similarities = np.dot(doc_vectors, query_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector))
    best_idx = np.argmax(similarities)
    return knowledge_base[best_idx]

# 5. 核心函数：生成最终回答
def generate_answer(user_query, context, history):
    history_text = "\n".join([f"用户：{h['user']}\nAI：{h['ai']}" for h in history])
    
    prompt = f"""请严格根据以下资料和对话历史回答问题。
    资料：{context}
    历史对话：{history_text}
    当前问题：{user_query}
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ==========================================
# 6. 开始多轮对话测试
# ==========================================
print("--- 多轮对话测试开始 ---")

# 第一轮
q1 = "迟到会被怎么惩罚？"
print(f"\n用户: {q1}")
c1 = retrieve(q1)
a1 = generate_answer(q1, c1, chat_history)
print(f"AI: {a1}")
# 存入记忆
chat_history.append({"user": q1, "ai": a1})

# 第二轮（关键测试点）
q2 = "那如果要请假呢？"  # 注意，这里问题很短，没有说“请假制度”
print(f"\n用户: {q2}")

# 步骤A：重写问题
new_q2 = rewrite_query(q2, chat_history)
print(f"【系统后台】发现带有上下文，自动把问题重写为: '{new_q2}'")

# 步骤B：用重写后的问题去检索
c2 = retrieve(new_q2)
print(f"【系统后台】检索到资料: {c2}")

# 步骤C：生成回答
a2 = generate_answer(new_q2, c2, chat_history)
print(f"AI: {a2}")
# 存入记忆
chat_history.append({"user": q2, "ai": a2})