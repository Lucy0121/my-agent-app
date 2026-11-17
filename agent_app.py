import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
# from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. 初始化网页设置
# ==========================================
st.set_page_config(page_title="超级企业助理", page_icon="🦸‍♂️")
st.title("🦸‍♂️ 超级企业 AI 助理 (Agent 版)")
st.caption("具备知识库检索、数学计算与多轮记忆能力")

with st.sidebar:
    # api_key = st.text_input("DeepSeek API Key", type="password")
    # 先尝试读取云端配置的 Key
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    
    # 如果没读到（比如在本地运行），就弹出一个输入框
    if not api_key:
        api_key = st.text_input("输入 API Key", type="password")

# ==========================================
# 2. 准备底层数据和 Embedding 模型
# ==========================================
@st.cache_resource
def load_data():
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    knowledge_base = [
        "考勤规定：迟到 3 次扣除全勤奖。",
        "请假规定：需提前一天在 OA 系统提交申请，病假需提供证明。",
        "年假规定：入职满一年有 5 天年假。",
        "差旅规定：出差每天补贴 100 元，高铁票需在月底前报销。"
    ]
    doc_vectors = model.encode(knowledge_base)
    return model, knowledge_base, doc_vectors

embed_model, knowledge_base, doc_vectors = load_data()

# ==========================================
# 3. 定义工具（Tools）
# ==========================================
# 注意看 @tool 装饰器：它会自动把函数名、参数、注释，变成我们昨天手写的那个 JSON 菜单！

@tool
def search_company_policy(query: str) -> str:
    """当你需要查询公司规章制度（如考勤、请假、年假、差旅报销）时，使用此工具。"""
    query_vector = embed_model.encode(query)
    similarities = np.dot(doc_vectors, query_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector))
    best_idx = np.argmax(similarities)
    return knowledge_base[best_idx]

@tool
def calculator(expression: str) -> str:
    """当你需要进行复杂的数学计算时（如加减乘除），使用此工具。传入具体的数学表达式字符串。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算失败: {e}"

# 把工具装进工具箱
tools = [search_company_policy, calculator]

# ==========================================
# 4. 网页交互逻辑
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 显示历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

if user_input := st.chat_input("例如：我出差高铁花了800，加上3天补贴一共多少钱？"):
    if not api_key:
        st.error("请先在左侧输入 API Key！")
    else:
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 初始化大模型（必须支持 tool_calling）
        llm = ChatOpenAI(api_key=api_key, base_url="https://api.deepseek.com", model="deepseek-chat", temperature=0)
        
        # 定义 Agent 的提示词（告诉它它是谁，以及它有哪些工具）
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个强大的企业助理。请根据员工的问题，合理使用你的工具。"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), # 这里用于记录 AI 调用工具的思考过程
        ])
        
        # 构建 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        # 执行器：verbose=True 会在终端打印它调用的工具过程
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # 执行回答
        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考并调用工具..."):
                try:
                    # 调用 Agent
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["output"]
                    st.markdown(answer)
                    
                    # 保存到历史记录
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"发生错误: {e}")