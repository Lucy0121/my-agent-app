import os

# 1. 在代码最前面强制指定国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface") # 可选，统一缓存路径

# 2. 从新包中导入 HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 3. 初始化时增加参数，强制从镜像下载
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},  # 指定使用CPU，避免MPS设备兼容问题
    encode_kwargs={'normalize_embeddings': True}, # 推荐，用于余弦相似度
    cache_folder=os.path.expanduser("~/.cache/huggingface/hub") # 缓存目录
)

# 【修改点 1】导入 PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 读 PDF 文件 & 切分
# ==========================================
print("正在读取并切分 PDF 文档...")
# 【修改点 2】换成 PyPDFLoader，并在括号里填上你的 pdf 文件名
loader = PyPDFLoader("./公司人字〔2020〕110号：员工考勤与休假管理制度.pdf") 
docs = loader.load()

# 切分代码完全不用变
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
print(f"PDF 被切分成了 {len(splits)} 个小段落。")

# ==========================================
# 2. 存入 Chroma 向量数据库
# ==========================================
# 【注意】如果你之前跑过 txt，为了避免数据冲突，建议把数据库目录换个名字，比如 "./chroma_db_pdf"
print("正在将文本转为向量并存入数据库...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_model,
    persist_directory="./chroma_db_pdf" # 换个新目录
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# 3. 定义大模型和提示词
# ==========================================
llm = ChatOpenAI(
    api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", 
    base_url="https://api.deepseek.com", # DeepSeek的接口地址
    model="deepseek-chat",
    streaming=True
)

template = """你是一个专业的企业内部助手。请严格根据以下检索到的资料，回答员工的问题。
资料：
{context}

员工问题：
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ==========================================
# 4. 拼装工作流
# ==========================================
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. 测试流式问答
# ==========================================
query = "请总结一下这份PDF文档的主要内容是什么？"
print(f"\n员工提问1：{query}")
print("\n--- AI 正在回答 ---")

query2 = "请问年假多久？婚假多久？产假多久？哺乳假多久？产假期间工资怎么核算？"
print(f"\n员工提问：{query2}")
print("\n--- AI 正在回答 ---")

for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)

print("\n\n--- 回答结束 ---")