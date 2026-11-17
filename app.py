import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================================
# 1. 网页初始化设置
# ==========================================
st.set_page_config(page_title="企业私有知识库", page_icon="🤖")
st.title("🤖 企业私有知识库 AI 助手")
st.caption("基于大模型与 RAG 架构（纯手工底层实现）")

# 侧边栏：输入 API Key
with st.sidebar:
    st.header("参数配置")
    api_key = st.text_input("DeepSeek API Key", type="password")
    st.markdown("---")
    st.markdown("💡 **提示**：本项目采用了 Query 重写与多轮对话记忆技术。")

# ==========================================
# 2. 核心模型与资料初始化（利用缓存只加载一次）
# ==========================================
@st.cache_resource
def load_models():
    # 加载 Embedding 模型
    embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    # 准备资料库
    knowledge_base = [
        "考勤规定：迟到 3 次扣除全勤奖。",
        "请假规定：需提前一天在 OA 系统提交申请，病假需提供证明。",
        "年假规定：入职满一年有 5 天年假。",
        "报销规定：高铁票需在月底前报销。"
    ]
    # 算好资料的向量
    doc_vectors = embed_model.encode(knowledge_base)
    return embed_model, knowledge_base, doc_vectors

embed_model, knowledge_base, doc_vectors = load_models()

# ==========================================
# 3. 核心 RAG 逻辑（完全照搬 raw_rag_memory.py）
# ==========================================
def retrieve(query, k=1):
    query_vector = embed_model.encode(query)
    similarities = np.dot(doc_vectors, query_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector))
    best_idx = np.argmax(similarities)
    return knowledge_base[best_idx]

def rewrite_query(user_query, history):
    if not history: return user_query
    
    # 【修改这里】适配 Streamlit 的 role 和 content 格式
    history_text = ""
    for h in history:
        if h["role"] == "user":
            history_text += f"用户：{h['content']}\n"
        elif h["role"] == "assistant":
            history_text += f"AI：{h['content']}\n"
            
    prompt = f"结合历史对话，把最新提问重写为一个完整独立的问题，只输出重写后的话。\n历史：{history_text}\n提问：{user_query}"
    res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content.strip()

# ==========================================
# 4. 网页交互：聊天记录界面
# ==========================================
# 初始化历史记录（Streamlit 的特性，必须用 session_state 保存状态）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 把历史记录显示在网页上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. 接收用户输入并回答
# ==========================================
if user_input := st.chat_input("请输入您的问题..."):
    if not api_key:
        st.error("请先在左侧输入 API Key！")
    else:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 显示用户提问
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 处理 AI 回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索并思考..."):
                # 1. 重写问题
                new_query = rewrite_query(user_input, st.session_state.messages)
                # 2. 检索资料
                context = retrieve(new_query)
                # 3. 生成回答
                # history_text = "\n".join([f"用户：{h['content']}" for h in st.session_state.messages])
                history_text = ""
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        history_text += f"用户：{msg['content']}\n"
                    else:
                        history_text += f"AI：{msg['content']}\n"
                prompt = f"根据资料：{context}\n历史对话：{history_text}\n回答问题：{user_input}"
                res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
                answer = res.choices[0].message.content
                
                # 显示回答
                st.markdown(answer)
        
        # 保存到记忆
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})