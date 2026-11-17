from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "这是第一句话。这是第二句话。这是第三句话。这里有一个极其极其极其极其极其极其极其长的一句话，长到超过了字数限制，所以它会被强行切开。"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20, 
    chunk_overlap=5,
    # 默认的切分符顺序：先找双换行、再找单换行、再找空格、再找句号，最后才按字切
    separators=["\n\n", "\n", "。", " ", ""] 
)

splits = text_splitter.split_text(text)

for i, s in enumerate(splits):
    print(f"【第 {i+1} 块】: {s}")