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


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser




# ==========================================
# 1. 读文件 & 切分（真实企业文档都很长）
# ==========================================
print("正在读取并切分文档...")
loader = TextLoader("./company_rules.txt", encoding="utf-8")
# loader = PDFlo("./公司人字〔2020〕110号：员工考勤与休假管理制度.pdf", encoding="utf-8")
docs = loader.load()

# 把长文档切成一段一段的（每段大概100个字，段与段之间重叠20字，防止意思被切断）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
print(f"文档被切分成了 {len(splits)} 个小段落。")

# ==========================================
# 2. 存入 Chroma 向量数据库（持久化）
# ==========================================
# 用我们昨天用的那个 BGE 模型来把文字变成向量
print("正在将文本转为向量并存入数据库...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 把切分好的段落存进 Chroma 数据库。这会在当前目录下生成一个 chroma_db 文件夹
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

# 把数据库变成检索器（以后直接查库，不用重复算）
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 找出最相关的2段话

# ==========================================
# 3. 定义大模型和提示词模板
# ==========================================
llm = ChatOpenAI(
    api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", 
    base_url="https://api.deepseek.com", # DeepSeek的接口地址
    model="deepseek-chat",
    streaming=True # 开启流式输出（一个字一个字蹦）
)

# 这个模板就是我们在 Day 29 写的字符串拼接，LangChain 把它标准化了
template = """你是一个专业的企业内部助手。请严格根据以下检索到的资料，回答员工的问题。
如果你不知道答案，就直接说“根据公司规定，我不知道”，千万不要编造。
资料：
{context}

员工问题：
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ==========================================
# 4. 拼装工作流（LangChain 的核心：LCEL 语法）
# ==========================================
# 这一串代码的意思是：把问题传进去 -> 检索出文档 -> 塞给模板 -> 交给大模型 -> 输出字符串
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. 测试流式问答
# ==========================================
query = "如果我迟到了3次会怎么样？另外出差住宿能住多少钱的？"
print(f"\n员工提问：{query}")
print("\n--- AI 正在回答 ---")

# 开启流式打印
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)

print("\n\n--- 回答结束 ---")