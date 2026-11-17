import numpy as np

# 1. 准备词向量表（还是昨天的坐标）
word_vectors = {
    "好": np.array([0.8, 0.9]),
    "开心": np.array([0.7, 0.85]),
    "差": np.array([-0.8, -0.9]),
    "环境": np.array([0.1, 0.2]) # 中性词，坐标靠中间
}

def sentence_to_vector(sentence):

    words = sentence.split(" ")
    print(f"当前句子分词：{words}")

    # 2. 把每个词的向量装进一个“矩阵”（表格）里
    # 这里的列表推导式就是把句子里的词对应的向量找出来
    vec_list = [word_vectors[w] for w in words if w in word_vectors]

    # 如果一句话里的词全都不认识，直接返回全 0
    if len(vec_list) == 0:
        return np.zeros(2)
        
    # 转换成 NumPy 的二维矩阵（这就是 AI 里的特征矩阵 X！）
    # shape 是 (单词数, 维度数)
    matrix = np.array(vec_list)
    print(f"这句话变成了一个 {matrix.shape[0]} 行 {matrix.shape[1]} 列的矩阵：\n{matrix}")

    # 3. 核心魔法：按列求平均值（axis=0 代表按列算）
    # 这样就把多行矩阵，压缩成了 1 行向量
    sentence_vec = np.mean(matrix, axis=0)
    return sentence_vec

# 测试一下
print("--- 测试第一句 ---")
vec1 = sentence_to_vector("好 开心")
print(f"最终这句话的特征向量: {vec1}\n")

print("--- 测试第二句 ---")
vec2 = sentence_to_vector("差 环境")
print(f"最终这句话的特征向量: {vec2}\n")