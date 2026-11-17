import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 消除刚才的 forked 警告

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 统一用新的 huggingface 包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 设置网页标题
st.title("📚 企业私有知识库问答助手")
st.caption("基于 LangChain + DeepSeek 构建")

# 侧边栏
with st.sidebar:
    st.header("设置")
    api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    st.markdown("---")
    st.markdown("### 关于本项目")
    st.markdown("这是一个基于 RAG 架构的智能问答助手，可以精准检索并回答公司制度。")

# 初始化 Session State (保存聊天记录)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 初始化模型和向量库 (只执行一次，提升速度)
@st.cache_resource
def init_rag():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    # 加载我们之前生成的本地数据库
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever

retriever = init_rag()

# 接收用户输入
if query := st.chat_input("请输入您的问题，例如：迟到会怎样？"):
    # 1. 把用户问题显示到界面
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. 调用大模型回答
    with st.chat_message("assistant"):
        if not api_key:
            st.error("请先在左侧边栏输入你的 API Key！")
        else:
            llm = ChatOpenAI(
                api_key=api_key, 
                base_url="https://api.deepseek.com", 
                model="deepseek-chat",
                streaming=True
            )
            
            template = """请严格根据以下资料回答问题。如果不知道，就说不知道。
            资料：{context}
            问题：{question}"""
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            
            # 使用 Streamlit 的流式输出组件
            response = st.write_stream(rag_chain.stream(query))
            
            # 保存回答到历史记录
            st.session_state.messages.append({"role": "assistant", "content": response})